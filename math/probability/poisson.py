#!/usr/bin/env python3
"""
Poisson distribution class
"""


class Poisson:
    """
    Represents a Poisson distribution.

    Attributes:
        lambtha (float): Expected number of occurrences in a given time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Args:
            data (list, optional): List of observed data points.
            lambtha (float, optional): Expected number of occurrences.

        Raises:
            ValueError: If lambtha <= 0 or data has fewer than 2 points.
            TypeError: If data is not a list.
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the PMF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value for k.
        """
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        # Manual exponential using Taylor series expansion
        def exp(x, terms=200):
            result = 1.0
            term = 1.0
            for n in range(1, terms):
                term *= x / n
                result += term
            return result

        # Manual factorial
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result

        value = (exp(-self.lambtha) * (self.lambtha ** k)) / factorial(k)
        return round(value, 10)
