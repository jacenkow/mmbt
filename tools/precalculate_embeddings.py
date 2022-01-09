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

"""A set of tools related to precalculating word embeddings."""

import argparse
from glob import glob
import json
import logging
import os
import pickle
import re

from fasttext import load_model
import lmdb
from mmf.utils.text import tokenize
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clean_text(text):
    """Clean text."""
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"//", " ", text)

    return text.lower()


def embeddings_lmdb(output_folder):
    """Open `.npy` embeddings and collect them into a single `lmdb` file."""
    embeddings = glob(os.path.join(output_folder, "*.npy"))

    assert len(embeddings) != 0, "Create first embeddings in .`npy` format."

    id_list = []
    env = lmdb.open(output_folder + ".lmdb", map_size=1099511627776)

    with env.begin(write=True) as txn:
        for infile in tqdm(embeddings):
            key = infile.split("/")[-1][:-4].encode()
            id_list.append(key)

            item = {"features": np.load(infile, allow_pickle=True)}
            txn.put(key, pickle.dumps(item))

        txn.put(b"keys", pickle.dumps(id_list))


def embeddings_npy(model_file, input_folder, output_folder):
    """Calculate embeddings from supplied JSON files. Save as `.npy` files."""
    # Create `--output_folder`.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the FastText model.
    logger.info("Loading FastText model")
    model = load_model(model_file)

    # Load JSON files.
    for i in glob(os.path.join(input_folder, "*.jsonl")):
        logger.info("Loading `jsonl`: " + i.split("/")[-1])

        with open(i, "r") as f:
            jsonl = [json.loads(i) for i in f.readlines()]

        for j in tqdm(jsonl):
            embedding = np.zeros((48, 200), dtype=np.float32)

            for index, token in enumerate(tokenize(_clean_text(j['text']))):
                embedding[index, :] = model.get_word_vector(token)

            np.save(os.path.join(output_folder, str(j['id']) + ".npy"),
                    embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        default=None,
        type=str,
        help="Location of BioWordVec embedding for fastText.",
    )
    parser.add_argument(
        "--input_folder", type=str, default=None, help="Input folder."
    )
    parser.add_argument(
        "--lmdb", type=bool, default=False, help="Convert to lmdb."
    )
    parser.add_argument(
        "--output_folder", type=str, default="./output", help="Output folder."
    )

    if not parser.parse_args().lmdb:
        embeddings_npy(parser.parse_args().model_file,
                       parser.parse_args().input_folder,
                       parser.parse_args().output_folder)

    if parser.parse_args().lmdb:
        embeddings_lmdb(parser.parse_args().output_folder)
