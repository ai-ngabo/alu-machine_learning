#!/usr/bin/env python3
"""Calculate correlation matrix from covariance matrix."""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix C.

    Args:
        C (numpy.ndarray): shape (d, d) covariance matrix

    Returns:
        numpy.ndarray: shape (d, d) correlation matrix

    Raises:
        TypeError: if C is not a numpy.ndarray
        ValueError: if C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    corr = np.zeros((d, d), dtype=float)
    # Standard deviations
    std = np.sqrt(np.diag(C))
    for i in range(d):
        for j in range(d):
            if std[i] == 0 or std[j] == 0:
                corr[i, j] = 0
            else:
                corr[i, j] = C[i, j] / (std[i] * std[j])
    return corr
