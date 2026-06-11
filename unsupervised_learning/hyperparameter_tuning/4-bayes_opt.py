#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization using a Gaussian Process."""

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Compute Expected Improvement."""
        mu, sigma2 = self.gp.predict(self.X_s)
        sigma = np.sqrt(sigma2)

        f_best = np.min(self.gp.Y)

        # Avoid division by zero
        sigma = np.where(sigma == 0, 1e-10, sigma)

        imp = f_best - mu - self.xsi
        Z = imp / sigma

        # Normal distribution functions (manual, no scipy)
        Phi = 0.5 * (1 + np.erf(Z / np.sqrt(2)))
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)

        EI = imp * Phi + sigma * phi
        EI = np.maximum(0, EI)

        X_next = self.X_s[np.argmax(EI)].reshape(-1)

        return X_next, EI
