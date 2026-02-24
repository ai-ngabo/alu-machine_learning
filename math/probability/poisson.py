#!/usr/bin/env python3
"""
Poisson distribution class
"""

from math import exp, factorial   # <-- move here


class Poisson:
    """
    Represents a Poisson distribution.

    Attributes:
        lambtha (float): Expected number of occurrences in a given time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        ...
    
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

        return (exp(-self.lambtha) * (self.lambtha ** k)) / factorial(k)
