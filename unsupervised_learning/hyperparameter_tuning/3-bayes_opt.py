#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm
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
        """Compute Expected Improvement acquisition function."""
        mu, sigma2 = self.gp.predict(self.X_s)
        sigma = np.sqrt(sigma2)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi

        sigma = np.where(sigma == 0, 1e-10, sigma)
        Z = imp / sigma

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI = np.maximum(0, EI)

        X_next = self.X_s[np.argmax(EI)].reshape(-1)

        return X_next, EI
