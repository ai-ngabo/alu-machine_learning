#!/usr/bin/env python3
"""Normalization module"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Args:
        X (np.ndarray): shape (d, nx)
        m (np.ndarray): mean of each feature (nx,)
        s (np.ndarray): std of each feature (nx,)

    Returns:
        normalized matrix of shape (d, nx)
    """

    return (X - m) / s
