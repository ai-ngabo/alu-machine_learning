#!/usr/bin/env python3
"""Moving average with bias correction"""

def moving_average(data, beta):
    """
    Calculates weighted moving average with bias correction

    Args:
        data (list): list of floats/ints
        beta (float): weight factor

    Returns:
        list: moving averages
    """

    v = 0
    result = []

    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        result.append(v_corrected)

    return result
