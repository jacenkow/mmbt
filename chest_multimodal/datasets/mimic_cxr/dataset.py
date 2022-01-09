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

from mmf.common.sample import Sample, SampleList
from mmf.datasets.mmf_dataset import MMFDataset
import torch


class MIMICCXRDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="mimic_cxr", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        self.report_field = "indication"

    def init_processors(self):
        super().init_processors()
        if self.config.use_images:
            self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        current_sample.id = torch.tensor(sample_info['id'], dtype=torch.int)
        current_sample.patient = torch.tensor(sample_info['patient'],
                                              dtype=torch.int)

        # Get text.
        try:
            processed_text = self.text_processor(
                {"text": sample_info[self.report_field]})
        except KeyError:
            processed_text = self.text_processor(
                {"text": sample_info['text']})
        current_sample.text = processed_text['text']

        if "input_ids" in processed_text:  # *BERT.
            current_sample.update(processed_text)

        # Get images.
        current_sample.image = self.image_db.from_path(
            sample_info['image'])['images'][0]

        # Get labels.
        current_sample.targets = torch.tensor(
            sample_info['labels'], dtype=torch.float
        )

        return current_sample

    @staticmethod
    def format_for_prediction(report):
        return generate_prediction(report)


class MIMICCXRFastTextDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="mimic_cxr", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

    def init_processors(self):
        super().init_processors()
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info['feature_path'] = f"{sample_info['id']}.npy"

        current_sample = Sample()
        current_sample.id = torch.tensor(sample_info['id'], dtype=torch.int)

        # Get pre-computed fastText embedding.
        features = self.features_db.get(sample_info)
        features_shape = self.config.processors.text_processor.params.max_length

        # It's not an image! MMF-naming convention...
        current_sample.text = features['image_feature_0'][:features_shape, :]

        # Get images.
        current_sample.image = self.image_db.from_path(
            sample_info['image'])['images'][0]

        # Get labels.
        current_sample.targets = torch.tensor(
            sample_info['labels'], dtype=torch.float
        )

        return current_sample

    @staticmethod
    def format_for_prediction(report):
        return generate_prediction(report)


def generate_prediction(report):
    scores = torch.nn.functional.sigmoid(report.scores)  # logits.
    labels = (scores > 0.5).float()  # Let's threshold at 0.5.

    predictions = []

    for index, study_id in enumerate(report.id[:-1]):
        probabilities = scores[index].tolist()
        label = labels[index].tolist()
        ground_truth = report.targets[index].tolist()

        predictions.append({
            "id": study_id.item(),
            "probabilities": probabilities,
            "labels": label,
            "ground_truth": ground_truth,
        })

    return predictions
