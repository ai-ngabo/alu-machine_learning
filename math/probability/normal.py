#!/usr/bin/env python3

class Normal:
    """Represents a normal (Gaussian) distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution
        """

        # If data is provided
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate standard deviation (population stddev)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

        else:
            # Use provided mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)
