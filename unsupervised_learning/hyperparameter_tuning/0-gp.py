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

        # Compute initial covariance matrix
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel
        """
        # Squared distance between each pair of points
        sqdist = (X1 - X2.T) ** 2

        # RBF kernel computation
        K = (self.sigma_f ** 2) * np.exp(-sqdist / (2 * self.l ** 2))

        return K
