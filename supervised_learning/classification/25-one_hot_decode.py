#!/usr/bin/env python3
"""
Contains the one_hot_decode function.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Parameters:
        one_hot (numpy.ndarray): One-hot encoded matrix with shape
                                 (classes, m).

    Returns:
        numpy.ndarray: 1D array containing numeric labels with shape (m,),
                       or None on failure.
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    # Verify that the input matrix is 2D and not empty
    if one_hot.ndim != 2 or one_hot.size == 0:
        return None

    try:
        # np.argmax along axis 0 extracts row index of each col
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
    