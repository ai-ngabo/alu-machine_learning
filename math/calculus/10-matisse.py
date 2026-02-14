#!/usr/bin/env python3

"""
This module provides a function to compute the derivative of a polynomial.

The function poly_derivative(poly) calculates the derivative of a polynomial
represented as a list of coefficients, where the index corresponds to the
power of x.

- If poly is not valid, returns None
- If the derivative is 0, returns [0]
- Otherwise returns a new list of coefficients representing the derivative
"""


def poly_derivative(poly):
    """Return the derivative of a polynomial given as a list of coefficients."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    derivative = [coef * idx for idx, coef in enumerate(poly)][1:]

    if all(c == 0 for c in derivative):
        return [0]

    return derivative
