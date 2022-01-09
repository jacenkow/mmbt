# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Grzegorz Jacenków.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Unimodal baselines."""

from mmf.common.registry import registry
from mmf.models import BaseModel, UnimodalBase, UnimodalText
from mmf.modules.embeddings import BiLSTMTextEmbedding
from mmf.utils.build import build_classifier_layer
import torch
from torch import nn
import torchvision
from transformers.modeling_bert import BertPredictionHeadTransform


@registry.register_model("unimodal_bert")
class UnimodalBERT(UnimodalText):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/models/unimodal/bert.yaml"

    def build(self):
        self.bert = UnimodalBase(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.config.text_encoder.params),
            nn.Linear(self.config.text_encoder.params.hidden_size,
                      self.config.num_labels),
        )

    def forward(self, sample_list):
        text = sample_list.input_ids
        args = [sample_list.input_mask, sample_list.segment_ids]

        pooled_output = self.bert(text, *args)
        logits = self.classifier(self.dropout(pooled_output))
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)

        return {"scores": reshaped_logits}


@registry.register_model("unimodal_gru")
class UnimodalGRU(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/models/unimodal/gru.yaml"

    def build(self):
        """GRU with BioWordVec embedding with no image input."""
        self.classifier = build_classifier_layer(self.config.classifier)
        self.text_embedding = self.build_text_embedding()

    def build_text_embedding(self):
        module_config = self.config.text_embedding

        return BiLSTMTextEmbedding(
            module_config.num_hidden,
            module_config.embedding_size,
            module_config.num_layers,
            module_config.dropout,
            module_config.bidirectional,
            module_config.rnn_type,
        )

    def forward(self, sample_list):
        text_features = self.text_embedding(sample_list['text'])
        logits = self.classifier(text_features)

        return {"scores": logits}


@registry.register_model("unimodal_resnet")
class UnimodalResNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/models/unimodal/resnet.yaml"

    def build(self):
        """ResNet-50 baseline for transfer learning."""
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features,
                                  self.config.num_labels)

    def forward(self, sample_list):
        logits = self.model(sample_list['image'])

        return {"scores": logits}