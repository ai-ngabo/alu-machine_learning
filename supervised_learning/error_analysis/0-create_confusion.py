#!/usr/bin/env python3
"""Creates a confusion matrix from labels and logits."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: one-hot numpy array of shape (m, classes)
        logits: one-hot numpy array of shape (m, classes)

    Returns:
        numpy array of shape (classes, classes)
    """
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)

    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes))

    np.add.at(confusion, (true, pred), 1)

    return confusion
