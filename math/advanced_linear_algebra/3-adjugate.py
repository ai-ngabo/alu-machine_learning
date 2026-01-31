#!/usr/bin/env python3

"""
Module for calculating the adjugate matrix of a square matrix.

This module provides a single function `adjugate(matrix)` that
validates input and computes the adjugate matrix.
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


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): input matrix

    Returns:
        list of lists: adjugate matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square or is empty
    """
    
    # --- Input validation ---
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # --- Compute cofactor matrix ---
    cofactors = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            submatrix = [
                r[:j] + r[j+1:]
                for k, r in enumerate(matrix)
                if k != i
            ]
            cofactor_val = ((-1) ** (i + j)) * determinant(submatrix)
            row_cofactors.append(cofactor_val)
        cofactors.append(row_cofactors)

    # --- Transpose cofactor matrix to get adjugate ---
    adj = [[cofactors[j][i] for j in range(n)] for i in range(n)]
    return adj
