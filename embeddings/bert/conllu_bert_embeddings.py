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
import re
import sys
import zipfile

import numpy as np

import bert_wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_conllu", type=str, help="Input CoNLL-U file")
    parser.add_argument("output_npz", type=str, help="Output NPZ file")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--casing", default=bert_wrapper.BertWrapper.CASING_UNCASED, help="Bert model casing")
    parser.add_argument("--language", default=bert_wrapper.BertWrapper.LANGUAGE_MULTILINGUAL, help="Bert model language")
    parser.add_argument("--layer_indices", default="-1,-2,-3,-4", type=str, help="Bert model layers to average")
    parser.add_argument("--size", default=bert_wrapper.BertWrapper.SIZE_BASE, help="Bert model size")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    parser.add_argument("--with_cls", default=False, action="store_true", help="Return also CLS embedding")
    args = parser.parse_args()
    args.layer_indices = list(map(int, args.layer_indices.split(",")))

    # Load CoNLL-U file
    sentences = []
    with open(args.input_conllu, mode="r", encoding="utf-8") as conllu_file:
        in_sentence = False
        for line in conllu_file:
            line = line.rstrip("\n")
            if line:
                if not in_sentence:
                    sentences.append([])
                    in_sentence = True
                if line.startswith("#"): continue
                columns = line.split("\t")
                assert len(columns) == 10
                sentences[-1].append(columns[1])
            else:
                in_sentence = False
    print("Loaded CoNLL-U file with {} sentences and {} words.".format(len(sentences), sum(map(len, sentences))), file=sys.stderr)

    bert = bert_wrapper.BertWrapper(language=args.language, size=args.size, casing=args.casing, layer_indices=args.layer_indices,
                                    with_cls=args.with_cls, threads=args.threads, batch_size=args.batch_size)
    with zipfile.ZipFile(args.output_npz, mode="w", compression=zipfile.ZIP_STORED) as output_npz:
        for i, embeddings in enumerate(bert.bert_embeddings(sentences)):
            if (i + 1) % 100 == 0: print("Processed {}/{} sentences.".format(i + 1, len(sentences)), file=sys.stderr)
            with output_npz.open("arr_{}".format(i), mode="w") as embeddings_file:
                np.save(embeddings_file, embeddings)
    print("Done, all embeddings saved.", file=sys.stderr)
