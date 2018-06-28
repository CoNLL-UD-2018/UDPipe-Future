#!/usr/bin/env python3
import pickle
import sys

import numpy as np

import fasttext

code = sys.argv[1]
source = sys.argv[2]

dictionary = fasttext.FastVector(vector_file=source, max_words=3000000)
with open("{}-conll17.words".format(code), mode="wb") as words_file:
    pickle.dump(dictionary.id2word, words_file)

np.save("{}-conll17.embeddings".format(code), dictionary.embed)
