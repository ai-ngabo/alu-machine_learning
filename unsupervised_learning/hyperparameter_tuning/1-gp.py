#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize Gaussian Process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Radial Basis Function kernel."""
        sqdist = (X1 - X2.T) ** 2
        return (self.sigma_f ** 2) * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """Predict mean and variance of Gaussian Process."""
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)

        return mu, sigma
