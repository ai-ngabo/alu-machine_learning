#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal (Gaussian) distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the Normal distribution."""
        
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)
