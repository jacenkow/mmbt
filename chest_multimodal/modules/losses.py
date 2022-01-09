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

from mmf.common.registry import registry
import torch
from torch import nn


@registry.register_loss("weighted_logit_bce")
class WeightedLogitBinaryCrossEntropy(nn.Module):
    """Weighted binary cross-entropy for MIMIC-CXR *official* split."""
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.weights = torch.tensor([0.795, 0.799, 0.951, 0.882, 0.969, 0.978,
                                     0.972, 0.781, 0.620, 0.765, 0.992, 0.934,
                                     0.954, 0.707]).to("cuda")

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = self.loss(scores, targets)

        return (loss * self.weights).mean() * targets.size(1)
