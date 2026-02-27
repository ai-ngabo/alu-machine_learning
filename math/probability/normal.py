#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""

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

            self.mean = float(sum(data) / len(data))

            variance = 0
            for x in data:
                variance += (x - self.mean) ** 2

            # ✅ POPULATION variance (divide by n)
            variance /= len(data)

            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the PDF for x"""

        pi = 3.141592653589793
        e = 2.718281828459045

        coeff = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)

        return coeff * (e ** exponent)

    def cdf(self, x):
        """Calculates the CDF for x"""

        e = 2.718281828459045

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        t = 1.0 / (1.0 + 0.3275911 * abs(z))

        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        erf = 1 - (
            (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
             * t * (e ** (-z * z)))
        )

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
