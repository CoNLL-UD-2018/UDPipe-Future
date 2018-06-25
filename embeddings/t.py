#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from fasttext import FastVector
sk_dictionary = FastVector(vector_file='wiki.sk.vec')

sk_dictionary.apply_transform('sk.txt')
import numpy as np
np.savez_compressed("sk.npz", embed=sk_dictionary.embed)
exit()

fr_dictionary = FastVector(vector_file='wiki.fr.vec')

fr_vector = fr_dictionary["chat"]
sk_vector = sk_dictionary["mačka"]
print(FastVector.cosine_similarity(fr_vector, sk_vector))
# Result should be 0.02

fr_dictionary.apply_transform('fr.txt')
sk_dictionary.apply_transform('sk.txt')

print(FastVector.cosine_similarity(fr_dictionary["chat"], sk_dictionary["mačka"]))
# Result should be 0.43
