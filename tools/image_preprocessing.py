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

from argparse import ArgumentParser
from glob import glob
from multiprocessing import cpu_count
import os
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
from skimage import color, io, transform, util

IMAGE_SIZE = (224, 224)


def _resize(filename, output):
    image = io.imread(filename)

    # Square the input image.
    if image.shape[0] > image.shape[1]:
        _diff = int((image.shape[0] - image.shape[1]) / 2)
        image = image[_diff:image.shape[0] - _diff, :]
    else:
        _diff = int((image.shape[1] - image.shape[0]) / 2)
        image = image[:, _diff:image.shape[1] - _diff]

    # Resize and cast to the RGB space.
    image = transform.resize(image, IMAGE_SIZE, preserve_range=True)
    image = color.gray2rgb(image).astype(np.uint8)

    io.imsave(output, image)
    print("Processed:", output)


def main(input_folder, output_folder, free_threads):
    """The master thread building the queue."""
    arguments = []

    # Add to the queue.
    for source in glob(os.path.join(input_folder, "p*", "p*", "s*", "*.jpg")):
        target = source.replace(input_folder, output_folder)

        # Make sure the target folder is created.
        Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)

        arguments.append((source, target))

    Parallel(n_jobs=cpu_count() - free_threads)(
        delayed(_resize)(i, j) for i, j in arguments)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--free_threads", type=int, default=None,
        help="Number of free threads.")
    parser.add_argument(
        "--input_folder", type=str, default=None, help="Input folder.")
    parser.add_argument(
        "--output_folder", type=str, default="./output", help="Output folder.")

    main(parser.parse_args().input_folder, parser.parse_args().output_folder,
         parser.parse_args().free_threads)
