#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):

        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Computes Expected Improvement and returns best next sample
        """

        # Predict GP mean and variance
        mu, sigma2 = self.gp.predict(self.X_s)

        sigma = np.sqrt(sigma2)

        # Current best observed value
        if self.minimize:
            f_best = np.min(self.gp.Y)
        else:
            f_best = np.max(self.gp.Y)

        # Avoid division by zero
        sigma = np.where(sigma == 0, 1e-10, sigma)

        # Compute improvement
        if self.minimize:
            imp = f_best - mu - self.xsi
        else:
            imp = mu - f_best - self.xsi

        Z = imp / sigma

        # Expected Improvement
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Ensure no negatives
        EI = np.maximum(0, EI)

        # Select next point
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
