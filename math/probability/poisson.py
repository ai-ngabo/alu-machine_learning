#!/usr/bin/env python3
"""
Poisson distribution class
"""

class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Parameters:
        - data: list of values to estimate lambtha (optional)
        - lambtha: expected number of occurrences (float, default=1.0)
        """
        if data is None:
            # Use provided lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Estimate lambtha as the mean of data
            self.lambtha = float(sum(data) / len(data))
