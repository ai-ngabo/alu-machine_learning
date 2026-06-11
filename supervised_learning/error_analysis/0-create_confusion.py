#!/usr/bin/env python3
"""Module to calculate a confusion matrix using numpy."""
import numpy as np


def create_confusion_matrix(labels, guesses):
    """
    Creates a confusion matrix.

    Args:
        labels: numpy.ndarray of shape (m,) containing the correct labels
        guesses: numpy.ndarray of shape (m,) containing predicted labels

    Returns:
        confusion: numpy.ndarray of shape (classes, classes) with the matrix
    """
    # Determine the number of unique classes dynamically
    num_classes = max(np.max(labels), np.max(guesses)) + 1

    confusion = np.zeros((num_classes, num_classes))

    # Populate matrix elements: rows are true labels, columns are guesses
    for true_idx, pred_idx in zip(labels, guesses):
        confusion[true_idx, pred_idx] += 1

    return confusion
