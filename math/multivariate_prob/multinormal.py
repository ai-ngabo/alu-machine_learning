#!/usr/bin/env python3
"""Multivariate Normal distribution module with PDF."""

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

        self.d = d  # number of dimensions
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)

        # Precompute constants for PDF
        self._cov_det = np.linalg.det(self.cov)
        self._cov_inv = np.linalg.inv(self.cov)
        self._norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * self._cov_det)

    def pdf(self, x):
        """
        Calculates the PDF at a data point x.

        Args:
            x (numpy.ndarray): shape (d, 1) data point

        Returns:
            float: PDF value at x

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        diff = x - self.mean
        exponent = -0.5 * (diff.T @ self._cov_inv @ diff)
        return float(self._norm_const * np.exp(exponent))
