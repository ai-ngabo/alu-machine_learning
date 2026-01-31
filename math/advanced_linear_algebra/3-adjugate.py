#!/usr/bin/env python3

"""
Module for calculating the minor matrix of a square matrix.

This module provides a single function `minor(matrix)` that
validates input and computes the minor matrix.
"""


def determinant(matrix):
    """Helper function to compute determinant recursively."""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)
    return det
def minor(matrix):

    """
    Calculate the minor matrix of a square matrix.

    Args:
        matrix (list of lists): input matrix

    Returns:
        list of lists: minor matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square or is empty
    """

    # --- Input validation ---
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # --- Compute minor matrix ---
    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            submatrix = [
                r[:j] + r[j+1:]
                for k, r in enumerate(matrix)
                if k != i
            ]
            row_minors.append(determinant(submatrix))
        minors.append(row_minors)

    return minors
