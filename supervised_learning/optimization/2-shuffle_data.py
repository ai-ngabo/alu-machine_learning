#!/usr/bin/env python3
"""Shuffle data module"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles data points in two matrices the same way

    Args:
        X (np.ndarray): shape (m, nx)
        Y (np.ndarray): shape (m, ny)

    Returns:
        X_shuffled, Y_shuffled
    """

    m = X.shape[0]
    permutation = np.random.permutation(m)

    return X[permutation], Y[permutation]
