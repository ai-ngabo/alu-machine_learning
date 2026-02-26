#!/usr/bin/env python3
"""Poisson distribution module"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor

        data: list of data to estimate lambtha
        lambtha: expected number of occurrences
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Estimate lambtha as mean of data
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for k successes
        """

        # Convert k to integer
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        # Compute factorial manually
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Compute e^(-lambtha) using approximation
        e = 2.7182818285
        pmf = (e ** (-self.lambtha)) * (self.lambtha ** k) / factorial

        return pmf
