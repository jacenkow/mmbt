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

"""Multimodal (image and text) approach taken from:

Towards Automated Diagnosis with Attentive Multi-modal Learning Using Electronic
Health Records and Chest X-Rays by (van Sonsbeek and Worring, 2020).

We tried to recreate the approach to the best of our knowledge following the
publication. Unfortunately, we were not able to run the official code.
"""

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.encoders import build_image_encoder
from mmf.modules.embeddings import BiLSTMTextEmbedding
from mmf.utils.build import build_classifier_layer
import torch
import torch.nn.functional as F
from torchnlp.nn import Attention


@registry.register_model("attentive")
class AttentiveMultimodal(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/models/attentive/defaults.yaml"

    def build(self):
        """ResNet50 with GRU and two attention blocks."""
        self.attention_a = Attention(self.config.attention.dimension)
        self.attention_b = Attention(self.config.attention.dimension)
        self.classifier = build_classifier_layer(self.config.classifier)
        self.image_encoder = build_image_encoder(self.config.image_encoder)
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
        image_features = self.image_encoder(sample_list['image'])
        text_features = self.text_embedding(sample_list['text'])[:, None, :]

        # Attention.
        image_attention, _ = self.attention_a(image_features, text_features)
        text_attention, _ = self.attention_b(text_features, image_features)

        image_features = F.relu(image_features + image_attention)
        text_features = F.relu(text_features + text_attention)

        # Fusion. Concatenate channels and `max` over them.
        combined, _ = torch.max(
            torch.cat([image_features, text_features], 1), 1)

        # Classifier.
        logits = self.classifier(combined)

        return {"scores": logits}
