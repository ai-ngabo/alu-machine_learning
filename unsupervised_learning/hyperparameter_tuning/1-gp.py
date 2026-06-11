#!/usr/bin/env python3
import numpy as np


class GaussianProcess:
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize a noiseless 1D Gaussian Process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        # Initial covariance matrix
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        RBF kernel
        """
        sqdist = (X1 - X2.T) ** 2
        return (self.sigma_f ** 2) * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """
        Predict mean and variance at new points X_s
        """
        # Kernel matrices
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        # Add small noise for numerical stability
        K_inv = np.linalg.inv(K)

        # Mean prediction
        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        # Variance prediction
        v = K_s.T @ K_inv
        cov = K_ss - v @ K_s
        sigma = np.diag(cov)

        return mu, sigma
