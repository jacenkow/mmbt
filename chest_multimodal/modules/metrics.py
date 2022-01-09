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

"""Label-based evaluation metrics, where we consider each class separately."""

from mmf.common.registry import registry
from mmf.modules.metrics import BaseMetric
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def _multilabel_confussion_matrix(scores, targets):
    """Multi-label confusion matrix (per label)."""
    true_positives = np.zeros(targets.shape[1])
    false_positives = np.zeros(targets.shape[1])
    true_negatives = np.zeros(targets.shape[1])
    false_negatives = np.zeros(targets.shape[1])

    for j in range(targets.shape[1]):  # Over labels.
        true_positives_label = 0
        false_positives_label = 0
        true_negatives_label = 0
        false_negatives_label = 0

        for i in range(targets.shape[0]):  # Over subjects.
            if int(targets[i, j]) == 1:
                if int(targets[i, j]) == 1 and int(scores[i, j]) == 1:
                    true_positives_label += 1
                else:
                    false_positives_label += 1
            else:
                if int(targets[i, j]) == 0 and int(scores[i, j]) == 0:
                    true_negatives_label += 1
                else:
                    false_negatives_label += 1

        true_positives[j] = true_positives_label
        false_positives[j] = false_positives_label
        true_negatives[j] = true_negatives_label
        false_negatives[j] = false_negatives_label

    return true_positives, false_positives, true_negatives, false_negatives


def _multilabel_micro_confussion_matrix(true_positives, false_positives,
                                        true_negatives, false_negatives):
    """Micro multi-label confusion matrix (per label)."""
    true_positives_micro = 0.0
    false_positives_micro = 0.0
    true_negatives_micro = 0.0
    false_negatives_micro = 0.0

    for i in range(len(true_positives)):
        true_positives_micro += true_positives[i]
        false_positives_micro += false_positives[i]
        true_negatives_micro += true_negatives[i]
        false_negatives_micro += false_negatives[i]

    return true_positives_micro, false_positives_micro, \
           true_negatives_micro, false_negatives_micro


@registry.register_metric("multilabel_accuracy")
class MultiLabelAccuracy(BaseMetric):
    """Metric for calculating multi-label accuracy."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_accuracy")
        self.threshold = kwargs.pop("threshold", 0.5)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # Threshold.
        scores = torch.sigmoid(scores)
        scores = (scores > self.threshold).float()

        accuracy = 0.0
        true_positives, false_positives, true_negatives, false_negatives = \
            _multilabel_confussion_matrix(scores.cpu().numpy(),
                                          targets.cpu().numpy())

        for i in range(len(true_positives)):
            accuracy += ((true_positives[i] + true_negatives[i]) /
                         (true_positives[i] + false_positives[i] +
                          true_negatives[i] + false_negatives[i]))

        accuracy /= len(true_positives)
        expected = sample_list["targets"]

        return expected.new_tensor(accuracy, dtype=torch.float)


@registry.register_metric("multilabel_macro_auroc")
class MultiLabelMacroAUROC(BaseMetric):
    """Metric for calculating multi-label AUROC based on *probabilities*."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_macro_auroc")
        self.average = kwargs.pop("average", "macro")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = torch.sigmoid(model_output["scores"]).cpu().numpy()
        targets = sample_list["targets"].cpu().numpy()
        auroc = 0.0

        if self.average == "macro":
            for j in range(targets.shape[1]):
                auroc += roc_auc_score(targets[:, j], scores[:, j])

            auroc /= targets.shape[1]
        elif self.average == "micro":
            auroc_per_class = np.zeros(targets.shape[1])
            counts_per_class = np.zeros(targets.shape[1])

            for j in range(targets.shape[1]):
                auroc_per_class[j] = roc_auc_score(targets[:, j], scores[:, j])
                counts_per_class[j] = np.sum(targets[:, j])

            auroc = np.multiply(auroc_per_class, counts_per_class) / \
                    np.sum(counts_per_class)
        else:
            raise ValueError("Wrong metric selected.")

        expected = sample_list["targets"]

        return expected.new_tensor(auroc, dtype=torch.float)


@registry.register_metric("multilabel_macro_f1")
class MultiLabelMacroF1(BaseMetric):
    """Metric for calculating multi-label macro F1."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_macro_f1")
        self.precision = MultiLabelMacroPrecision(*args, **kwargs)
        self.recall = MultiLabelMacroRecall(*args, **kwargs)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        precision = self.precision.calculate(sample_list, model_output)
        recall = self.recall.calculate(sample_list, model_output)
        f1 = 2 * (precision * recall) / (precision + recall)

        expected = sample_list["targets"]

        return expected.new_tensor(f1, dtype=torch.float)


@registry.register_metric("multilabel_macro_precision")
class MultiLabelMacroPrecision(BaseMetric):
    """Metric for calculating multi-label macro precision."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_macro_accuracy")
        self.average = kwargs.pop("average", "macro")
        self.threshold = kwargs.pop("threshold", 0.5)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # Threshold.
        scores = torch.sigmoid(scores)
        scores = (scores > self.threshold).float()

        precision = 0.0
        true_positives, false_positives, true_negatives, false_negatives = \
            _multilabel_confussion_matrix(scores.cpu().numpy(),
                                          targets.cpu().numpy())

        if self.average == "macro":
            for i in range(len(true_positives)):
                if true_positives[i] + false_positives[i]:
                    precision += (true_positives[i] /
                                  (true_positives[i] + false_positives[i]))

            precision /= len(true_positives)
        elif self.average == "micro":
            true_positives_micro, false_positives_micro, _, _ = \
                _multilabel_micro_confussion_matrix(true_positives,
                                                    false_positives,
                                                    true_negatives,
                                                    false_negatives)

            if true_positives_micro + false_positives_micro:
                precision = true_positives_micro / \
                            (true_positives_micro + false_positives_micro)
        else:
            raise ValueError("Wrong metric selected.")

        expected = sample_list["targets"]

        return expected.new_tensor(precision, dtype=torch.float)


@registry.register_metric("multilabel_macro_recall")
class MultiLabelMacroRecall(BaseMetric):
    """Metric for calculating multi-label macro recall."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_macro_accuracy")
        self.average = kwargs.pop("average", "macro")
        self.threshold = kwargs.pop("threshold", 0.5)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # Threshold.
        scores = torch.sigmoid(scores)
        scores = (scores > self.threshold).float()

        recall = 0.0
        true_positives, false_positives, true_negatives, false_negatives = \
            _multilabel_confussion_matrix(scores.cpu().numpy(),
                                          targets.cpu().numpy())

        if self.average == "macro":
            for i in range(len(true_positives)):
                if true_positives[i] + false_negatives[i]:
                    recall += (true_positives[i] /
                               (true_positives[i] + false_negatives[i]))

            recall /= len(true_positives)
        elif self.average == "micro":
            true_positives_micro, _, _, false_negatives_micro = \
                _multilabel_micro_confussion_matrix(true_positives,
                                                    false_positives,
                                                    true_negatives,
                                                    false_negatives)
            if true_positives_micro + false_negatives_micro:
                recall = true_positives_micro / \
                         (true_positives_micro + false_negatives_micro)
        else:
            raise ValueError("Wrong metric selected.")

        expected = sample_list["targets"]

        return expected.new_tensor(recall, dtype=torch.float)


@registry.register_metric("multilabel_micro_auroc")
class MultiLabelMicroAUROC(MultiLabelMacroAUROC):
    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_auroc"


@registry.register_metric("multilabel_micro_f1")
class MultiLabelMicroF1(BaseMetric):
    """Metric for calculating multi-label micro F1."""
    def __init__(self, *args, **kwargs):
        super().__init__("multilabel_micro_f1")
        self.precision = MultiLabelMicroPrecision(*args, **kwargs)
        self.recall = MultiLabelMicroRecall(*args, **kwargs)

    def calculate(self, sample_list, model_output, *args, **kwargs):
        precision = self.precision.calculate(sample_list, model_output)
        recall = self.recall.calculate(sample_list, model_output)
        f1 = 2 * (precision * recall) / (precision + recall)

        expected = sample_list["targets"]

        return expected.new_tensor(f1, dtype=torch.float)


@registry.register_metric("multilabel_micro_precision")
class MultiLabelMicroPrecision(MultiLabelMacroPrecision):
    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_precision"


@registry.register_metric("multilabel_micro_recall")
class MultiLabelMicroRecall(MultiLabelMacroRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_recall"
