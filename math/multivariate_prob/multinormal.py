#!/usr/bin/env python3
"""Multivariate Normal distribution module."""

import numpy as np


class MultiNormal:
    """Represents a multivariate normal distribution."""

    def __init__(self, data):
        """
        Initialize a Multivariate Normal distribution.

        Args:
            data (numpy.ndarray): shape (d, n) 

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if n < 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Compute mean (d,1)
        self.mean = np.mean(data, axis=1).reshape(d, 1)

        # Center the data
        data_centered = data - self.mean

        # Compute covariance manually: (X_centered @ X_centered.T) / (n - 1)
        self.cov = (data_centered @ data_centered.T) / (n - 1)
