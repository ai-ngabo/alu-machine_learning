#!/usr/bin/env python3
"""Binomial distribution module."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize the Binomial distribution."""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean_data = sum(data) / len(data)
            variance_data = sum((x - mean_data) ** 2 for x in data) / len(data)

            n_est = round((mean_data ** 2) / (mean_data - variance_data))
            self.n = n_est
            self.p = float(mean_data / n_est)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes k."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        # Compute combination nCk
        comb = 1
        for i in range(1, k + 1):
            comb *= (self.n - i + 1) / i
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes k."""
        k = int(k)
        total = 0
        for i in range(0, k + 1):
            total += self.pmf(i)
        return total
