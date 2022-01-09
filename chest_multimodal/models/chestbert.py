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

"""We use MMBT implementation from Transformers."""

from copy import deepcopy
from dataclasses import dataclass
import logging
import random
from typing import Dict, Optional, Union

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.encoders import (
    EncoderFactory,
    ImageEncoderFactory,
    ImageEncoderTypes,
    MultiModalEncoderBase,
    TextEncoderFactory,
    TextEncoderTypes,
    TorchvisionResNetImageEncoder,
    TransformerEncoder,
)
from mmf.modules.hf_layers import replace_with_jit
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.modeling import get_bert_configured_parameters
import numpy as np
from omegaconf import II, DictConfig, OmegaConf
import torch
from torch import Tensor, nn
from torch.autograd import Function
from transformers.modeling_bert import (
    BertForPreTraining,
    BertPredictionHeadTransform,
)

from chest_multimodal.models.mmbt import MMBTConfig, MMBTModel
from chest_multimodal.modules.losses import BarlowTwinsLoss

logger = logging.getLogger(__name__)


class MMBTResNet(MultiModalEncoderBase):
    """ResNet-50 interface for MMBT."""
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        replace_with_jit()

    def build(self):
        encoders = self._build_encoders(self.config)
        text_encoder, modal_encoder = encoders[0], encoders[1]

        self._encoder_config = text_encoder.config
        self._mmbt_config = MMBTConfig(
            self._encoder_config,
            num_labels=self.config.num_labels,
            modal_hidden_size=self.config.modal_hidden_size
        )

        self.use_modal_start_token = self.config.use_modal_start_token
        self.use_modal_end_token = self.config.use_modal_end_token
        self.num_max_segment = self.config.text_encoder.params.get(
            "num_segments", 2)

        self.mmbt = MMBTModel(self._mmbt_config, text_encoder, modal_encoder)

    def extract_modal_end_token(self, sample_list: Dict[str, Tensor]):
        batch_size = sample_list["input_ids"].size(0)
        device = sample_list["input_ids"].device
        gather_index = sample_list["input_mask"].sum(1, keepdim=True) - 1

        modal_end_token = (
            torch.gather(sample_list["input_ids"], 1, gather_index)
            .squeeze(1)
            .clone()
            .detach()
        )

        sample_list["input_ids"] = torch.cat(
            [sample_list["input_ids"][:, 1:],
             sample_list["input_ids"][:, -1:]], dim=1
        )

        sample_list["input_mask"] = torch.cat([
            sample_list["input_mask"][:, 1:],
            torch.zeros([batch_size, 1], dtype=torch.long, device=device),
        ], dim=1)

        return modal_end_token

    def forward(self, sample_list: Dict[str, Tensor]):
        input_modal = sample_list["image"]

        modal_start_token: Optional[Tensor] = None
        if self.use_modal_start_token:
            modal_start_token = sample_list["input_ids"][:, 0].clone().detach()

        modal_end_token: Optional[Tensor] = None
        if self.use_modal_end_token:
            modal_end_token = self.extract_modal_end_token(sample_list)

        if "modal_token_type_ids" in sample_list:
            modal_token_type_ids = sample_list["modal_token_type_ids"]
        else:
            token_value = 0
            segment_ids = sample_list["segment_ids"]
            max_id = segment_ids.max()
            min_id = segment_ids.min()

            if max_id == min_id:
                if max_id == torch.tensor(0, dtype=max_id.dtype):
                    token_value = 1
            else:
                max_segment = self.num_max_segment - 1

                if max_id != torch.tensor(max_segment, dtype=max_id.dtype):
                    token_value = max_segment

            modal_token_type_ids = torch.full(
                (input_modal.size(0), 1),
                fill_value=token_value,
                dtype=torch.long,
                device=input_modal.device,
            )

        if input_modal.dim() == 2:
            input_modal = input_modal.unsqueeze(dim=1)

        output = self.mmbt(
            input_modal,
            input_ids=sample_list["input_ids"],
            modal_start_tokens=modal_start_token,
            modal_end_tokens=modal_end_token,
            attention_mask=sample_list["input_mask"],
            token_type_ids=sample_list["segment_ids"],
            modal_token_type_ids=modal_token_type_ids,
            position_ids=None,
            modal_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=None,
        )

        return output


class ChestBERTClassification(nn.Module):
    """ChestBERT with BertPredictionHeadTransform head."""
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTResNet(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config
        self.num_labels = self.config.num_labels

        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.encoder_config),
            nn.Linear(self.encoder_config.hidden_size, self.config.num_labels),
        )

    def forward(self, sample_list: Dict[str, Tensor]):
        pooled_output = self.dropout(self.bert(sample_list)[1])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)

        return {'scores': reshaped_logits}


@registry.register_model("chestbert")
class ChestBERT(BaseModel):
    """ChestBERT end-point for the MMF (based on Facebook's MMBT)."""
    @dataclass
    class Config(BaseModel.Config):
        bert_model_name: str = "bert-base-uncased"
        direct_features_input: bool = False
        finetune_lr_multiplier: float = 1
        freeze_text: bool = False
        freeze_modal: bool = False
        freeze_complete_base: bool = False
        fused_feature_only: bool = False
        modal_hidden_size: int = 2048
        model: str = "chestbert"
        num_labels: int = 14
        output_dim: int = 768
        text_hidden_size: int = 768
        use_modal_start_token: bool = True
        use_modal_end_token: bool = True

        # Encoders.
        modal_encoder: EncoderFactory.Config = ImageEncoderFactory.Config(
            type=ImageEncoderTypes.torchvision_resnet,
            params=TorchvisionResNetImageEncoder.Config()
        )
        text_encoder: EncoderFactory.Config = TextEncoderFactory.Config(
            type=TextEncoderTypes.transformer,
            params=TransformerEncoder.Config(
                bert_model_name=II("bert_model_name")),
        )

    def __init__(self, config: Union[DictConfig, Config], *args, **kwargs):
        super().__init__(config)

    def build(self):
        self.model = ChestBERTClassification(self.config)

        if self.config.freeze_complete_base or self.config.freeze_text:
            for p in self.model.bert.mmbt.transformer.parameters():
                p.requires_grad = False

        if self.config.freeze_complete_base or self.config.freeze_modal:
            for p in self.model.bert.mmbt.modal_encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        model = super().from_pretrained(model_name, *args, **kwargs)
        config = load_pretrained_model(model_name)["full_config"]
        OmegaConf.set_struct(config, True)

        return model

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/models/mmbt/defaults.yaml"

    def forward(self, sample_list: Dict[str, Tensor]):
        return self.model(sample_list)

    def get_optimizer_parameters_for_bert(self, config):
        lr = config.optimizer.params.lr
        model_config = config.model_config.get(config.model, {})
        finetune_lr_multiplier = model_config.get("finetune_lr_multiplier", 1)

        parameters = []
        # FIXME: MMF has a bug here. Can return an empty params list.
        for name, submodule in self.model.named_children():
            if name == "classifier":
                continue
            parameters += get_bert_configured_parameters(
                submodule, lr * finetune_lr_multiplier
            )
            logger.info(f"Overriding {name} module's LR to "
                        f"{lr * finetune_lr_multiplier}")

        parameters += get_bert_configured_parameters(self.model.classifier)

        return parameters
