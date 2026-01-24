#!/usr/bin/env python3
"""
Defines a function cat_matrices2D that
 concatenates two 2D matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along the given axis.
    Return a new matrix, or None if they cannot be concatenated.
    """
    # Concatenate along rows
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Concatenate along columns
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    # Invalid axis
    return None
