#!/usr/bin/env python3

"""
Module for calculating the determinant of a matrix.

This module provides a single function `determinant(matrix)` that
validates input and computes the determinant recursively.
"""

def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
        matrix (list of lists): input matrix

    Returns:
        int/float: determinant of the matrix

    Raises:
        TypeError: if matrix is not a list of lists
        ValueError: if matrix is not square
    """
    # --- Input validation ---
    if (not isinstance(matrix, list) or
            any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix case
    if matrix == [[]]:
        return 1

    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")

    # Check square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # --- Base cases ---
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # --- Recursive case (Laplace expansion along first row) ---
    det = 0
    for j in range(n):
        # Build minor by removing row 0 and column j
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det
