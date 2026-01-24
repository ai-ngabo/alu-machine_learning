#!/usr/bin/env python3
"""
Defines a function add_matrices2D that adds two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """new matrix with sum of mat1 and mat2, or None if shapes differ."""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
