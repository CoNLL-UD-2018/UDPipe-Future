#!/usr/bin/env python3
import pickle
import sys

import numpy as np

import fasttext

code = sys.argv[1]

dictionary = fasttext.FastVector(vector_file="wiki.{}.vec".format(code), max_words=1000000)
with open("{}.words".format(code), mode="wb") as words_file:
    pickle.dump(dictionary.id2word, words_file)

np.save("{}.embeddings".format(code), dictionary.embed)

try:
    dictionary.apply_transform("{}.txt".format(code))
    np.save("{}.shared-embeddings".format(code), dictionary.embed)
except:
    pass
