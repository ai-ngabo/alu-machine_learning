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
        """Calculate z-score"""
        return (x - self.mean) / self.stddev

    def cdf(self, x):
        """Calculate the cumulative distribution function"""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Constants for approximation
        a1 = 0.0705230784
        a2 = 0.0422820123
        a3 = 0.0092705272
        a4 = 0.0001520143
        a5 = 0.0002765672
        a6 = 0.0000430638

        abs_z = z if z >= 0 else -z

        t = 1 + a1 * abs_z \
            + a2 * abs_z ** 2 \
            + a3 * abs_z ** 3 \
            + a4 * abs_z ** 4 \
            + a5 * abs_z ** 5 \
            + a6 * abs_z ** 6

        erf = 1 - (1 / (t ** 16))

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
