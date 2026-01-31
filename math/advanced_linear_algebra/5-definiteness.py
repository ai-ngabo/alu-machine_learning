#!/usr/bin/env python3
"""
Module for calculating the definiteness of a square matrix.

This module provides a single function `definiteness(matrix)` that
validates input and computes whether the matrix is positive definite,
positive semi-definite, negative semi-definite, negative definite,
or indefinite.
"""


def definiteness(matrix):
    """
    Calculate the definiteness of a square matrix.

    Args:
        matrix (numpy.ndarray): input matrix of shape (n, n)

    Returns:
        str: one of "Positive definite", "Positive semi-definite",
             "Negative semi-definite", "Negative definite", "Indefinite",
             or None if not valid

    Raises:
        TypeError: if matrix is not a numpy.ndarray
    """
    # --- Input validation ---
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # --- Symmetry check ---
    if not np.allclose(matrix, matrix.T):
        return None

    # --- Compute eigenvalues ---
    try:
        eigvals = np.linalg.eigvals(matrix)
    except Exception:
        return None

    
    tol = 1e-10
    pos = np.all(eigvals > tol)
    neg = np.all(eigvals < -tol)
    semi_pos = np.all(eigvals >= -tol) and np.any(np.abs(eigvals) <= tol)
    semi_neg = np.all(eigvals <= tol) and np.any(np.abs(eigvals) <= tol)

    
    if pos:
        return "Positive definite"
    if semi_pos:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if semi_neg:
        return "Negative semi-definite"
    if np.any(eigvals > tol) and np.any(eigvals < -tol):
        return "Indefinite"

    return None
