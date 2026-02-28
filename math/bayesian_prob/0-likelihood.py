#!/usr/bin/env python3
"""
Module that computes the likelihood of observing data
given a set of hypothetical probabilities.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining x successes
    out of n trials for each probability in P.

    Parameters:
    x (int): number of successes
    n (int): number of trials
    P (np.ndarray): 1D array of hypothetical probabilities

    Returns:
    np.ndarray: likelihood values for each probability in P
    """

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
                "x must be an integer that is greater than or equal to 0"
                )

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient
    factorial = np.math.factorial
    coeff = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood calculation
    likelihoods = coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
