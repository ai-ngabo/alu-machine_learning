#!/usr/bin/env python3

"""
This module provides a function to compute the summation of squares.

The function summation_i_squared(n) calculates:
    sum_{i=1}^n i^2

- n is the stopping condition
- Returns the integer value of the sum
- If n is not a valid number, returns None
- No loops are used; the closed-form formula is applied
"""

def summation_i_squared(n):

    """Return the sum of squares from 1 to n, or None if invalid."""
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
