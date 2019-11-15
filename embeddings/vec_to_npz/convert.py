#!/usr/bin/env python3

# This file is part of UDPipe Future <http://github.com/CoNLL-UD-2018/UDPipe-Future>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vec_file", type=str, help="Input .vec file path")
parser.add_argument("npz_file", type=str, help="Output .npz file path")
parser.add_argument("--max_words", default=None, type=int, help="Maximum number of words to save")
args = parser.parse_args()

with open(args.vec_file, "r", encoding="utf-8") as vec_file:
    num_words, dim = map(int, vec_file.readline().rstrip().split())

    if args.max_words is not None:
        num_words = min(num_words, args.max_words)

    words = np.empty(num_words, dtype=np.object)
    embeddings = np.empty((num_words, dim), dtype=np.float32)

    for i, line in enumerate(vec_file):
        if i >= num_words:
            break

        columns = line.rstrip("\n").split()
        words[i] = columns[0]
        embeddings[i] = columns[1:1+dim]


np.savez(args.npz_file, words=words, embeddings=embeddings)
