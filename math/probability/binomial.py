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

            # Estimate p from data
            mean_data = sum(data) / len(data)
            p_est = mean_data / max(data)  # first estimate p
            # Estimate n
            variance_data = sum((x - mean_data) ** 2 for x in data) / len(data)
            n_est = round(mean_data / p_est)
            # Recalculate p based on rounded n
            self.n = n_est
            self.p = float(mean_data / n_est)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
