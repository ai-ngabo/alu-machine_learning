#!/usr/bin/env python3
import numpy as np
"""np_cat that concatenates two along a specific axis."""


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy.ndarrays along the given axis."""
    return np.concatenate((mat1, mat2), axis=axis)
