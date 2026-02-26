#!/usr/bin/env python3
"""
Poisson distribution module
"""


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
        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        # Manual factorial
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Manual exponential using Taylor series expansion
        def exp(x, terms=200):
            result = 1.0
            term = 1.0
            for n in range(1, terms):
                term *= x / n
                result += term
            return result

        value = (exp(-self.lambtha) * (self.lambtha ** k)) / factorial
        return float("{:.10f}".format(value))

    def cdf(self, k):
        """
        Calculates the CDF for k successes
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

        # Sum PMF from 0 to k
        cdf_value = 0.0
        for i in range(0, k + 1):
            cdf_value += (exp(-self.lambtha) * (self.lambtha ** i)) / factorial(i)

        return cdf_value
