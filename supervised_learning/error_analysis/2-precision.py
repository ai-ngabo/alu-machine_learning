#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """Computes precision for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    true_positive = np.diag(confusion)
    predicted_positive = np.sum(confusion, axis=0)

    return true_positive / predicted_positive
