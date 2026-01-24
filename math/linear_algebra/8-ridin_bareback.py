#!/usr/bin/env python3
"""
Defines a function mat_mul that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices element-wise.
    Return a new matrix, or None if they cannot be multiplied.
    """
    # Check if multiplication is possible: columns of mat1 == rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Perform multiplication
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            val = 0
            for k in range(len(mat2)):
                val += mat1[i][k] * mat2[k][j]
            row.append(val)
        result.append(row)
    return result
