#!/usr/bin/env python3
import numpy as np


def sensitivity(confusion):
    """
    Calculates sensitivity (recall) for each class.

    Args:
        confusion: numpy array of shape (classes, classes)

    Returns:
        numpy array of shape (classes,)
    """
    true_positive = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positive / actual_positives
