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

import os

from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path

from chest_multimodal.datasets.mimic_cxr.dataset import (
    MIMICCXRDataset,
    MIMICCXRFastTextDataset,
)


@registry.register_builder("mimic_cxr")
class MIMICCXRBuilder(MMFDatasetBuilder):
    def __init__(self,
                 dataset_name="mimic_cxr",
                 dataset_class=MIMICCXRDataset,
                 *args, **kwargs):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = MIMICCXRDataset

    @classmethod
    def config_path(cls):
        return "chest_multimodal/configs/datasets/mimic_cxr/defaults.yaml"

    def build(self, config, *args, **kwargs):
        test_path = get_absolute_path(
            os.path.join(
                get_mmf_env(key="data_dir"),
                "mimic_cxr", "annotations", "train.jsonl",
            )
        )

        assert PathManager.exists(test_path), (
            "The MIMIC-CXR dataset has not been found in the system. " +
            "Follow the instructions on the GitHub page."
        )

        super().build(config, *args, **kwargs)

    def load(self, config, dataset, *args, **kwargs):
        if config.use_fasttext:
            self.dataset_class = MIMICCXRFastTextDataset

        self.dataset = super().load(config, dataset, *args, **kwargs)

        return self.dataset
