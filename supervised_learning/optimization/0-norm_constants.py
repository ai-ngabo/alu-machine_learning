#!/usr/bin/env python3
"""Normalization constants module"""

import numpy as np


def normalization_constants(X):
    """
    Calculates normalization constants of a matrix

    Args:
        X (np.ndarray): shape (m, nx)

    Returns:
        mean, std: per-feature mean and standard deviation
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
