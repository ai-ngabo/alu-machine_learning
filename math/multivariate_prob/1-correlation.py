#!/usr/bin/env python3
"""Calculate correlation matrix from covariance matrix."""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix C.

    Args:
        C (numpy.ndarray): shape (d, d)
