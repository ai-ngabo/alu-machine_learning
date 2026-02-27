#!/usr/bin/env python3
"""Calculate mean and covariance of a dataset."""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance matrix of a dataset X.

    Args:
        X (numpy.ndarray): shape (n, d) containing n data points of d dimensions

    Returns:
        mean (numpy.ndarray): shape (1, d) containing the mean
        cov (numpy.ndarray): shape (d, d) containing the covariance matrix

    Raises:
        TypeError: if X is not a 2D numpy.ndarray
        ValueError: if n < 2
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Compute mean
    mean = np.mean(X, axis=0).reshape(1, d)

    # Center the data
    X_centered = X - mean

    # Compute covariance manually: (X_centered.T @ X_centered) / (n - 1)
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
