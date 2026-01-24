#!/usr/bin/env python3
"""
Defines a function np_cats.
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy.ndarrays along the given axis."""
    return np.concatenate((mat1, mat2), axis=axis)
