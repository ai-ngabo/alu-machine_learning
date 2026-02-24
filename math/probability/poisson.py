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
