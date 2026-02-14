#!/usr/bin/env python3
"""
This module provides a function to compute the integral of a polynomial.

The function poly_integral(poly, C=0) calculates the integral of a polynomial
represented as a list of coefficients, where the index corresponds to the
power of x.

- poly is a list of coefficients
- C is an integer representing the integration constant
- If poly or C are not valid, returns None
- If a coefficient is a whole number, it is represented as an integer
- Returns a new list of coefficients representing the integral
- The returned list is as small as possible
"""


def poly_integral(poly, C=0):
    """
    Return the integral of a polynomial.

    The polynomial is given as a list of coefficients, where the index
    represents the power of x. The integration constant C is added as
    the first element of the result.
    """
    # Validate inputs
    if not isinstance(poly, list) or not all(
        isinstance(c, (int, float)) for c in poly
    ):
        return None
    if not isinstance(C, (int, float)):
        return None

    # Compute integral coefficients
    integral = [C]
    for idx, coef in enumerate(poly):
        new_coef = coef / (idx + 1)
        # Represent whole numbers as int
        if float(new_coef).is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    # Remove trailing zeros to keep list minimal
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
