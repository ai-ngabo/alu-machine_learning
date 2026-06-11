#!/usr/bin/env python3
"""Calculates specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Computes specificity for each class.

    Args:
        confusion: numpy.ndarray of shape (classes, classes)

    Returns:
        numpy.ndarray of shape (classes,)
    """
    total = np.sum(confusion, axis=1).sum()
    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, axis=0) - true_positive
    false_negative = np.sum(confusion, axis=1) - true_positive
    true_negative = total - (true_positive + false_positive + false_negative)

    return true_negative / (true_negative + false_positive)
