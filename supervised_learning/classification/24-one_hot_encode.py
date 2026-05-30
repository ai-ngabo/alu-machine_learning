#!/usr/bin/env python3
"""
Contains the one_hot_encode function.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
        Y (numpy.ndarray): Numeric class labels with shape (m,).
        classes (int): The total number of classes.

    Returns:
        numpy.ndarray: One-hot encoding matrix with shape (classes, m),
                       or None on failure.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

    if classes <= 0 or Y.ndim != 1 or Y.size == 0:
        return None

    # Ensure all class labels are valid and within bounds
    if np.min(Y) < 0 or np.max(Y) >= classes:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        # Advanced indexing: pairs row indices from Y 
        one_hot[Y, np.arange(m)] = 1.0
        return one_hot
    except Exception:
        return None
