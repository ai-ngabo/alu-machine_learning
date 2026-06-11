#!/usr/bin/env python3
"""Calculates sensitivity (recall) for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Computes sensitivity for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    true_positive = np.diag(confusion)
    actual = np.sum(confusion, axis=1)

    return true_positive / actual
