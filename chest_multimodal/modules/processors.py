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
from mmf.datasets.processors import MaskedTokenProcessor, FastTextProcessor
from mmf.utils.configuration import get_mmf_env
from mmf.utils.file_io import PathManager


@registry.register_processor("biowordvec")
class BioWordVecProcessor(FastTextProcessor):
    def __call__(self, item):
        self.model_file = os.path.join(get_mmf_env(key="data_dir"),
                                       self.config.model_file)

        assert PathManager.exists(self.model_file), (
                "BioWordVec has not been found. Download the model."
        )

        self._load_fasttext_model(self.model_file)

        return super().__call__(item)
