#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)

            variance = 0
            for value in data:
                variance += (value - self.mean) ** 2

            variance /= len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculate the z-score of x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a z-score"""
        return z * self.stddev + self.mean

    def cdf(self, x):
        """Calculate the cumulative distribution function"""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1
        if z < 0:
            sign = -1
        z = abs(z)

        t = 1 / (1 + p * z)
        erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
                   * t * (2.718281828459045 ** (-z ** 2)))

        erf *= sign

        return 0.5 * (1 + erf)
