#!/usr/bin/env python3

"""Module for calculating the shape of a matrix."""

def matrix_shape(matrix):

    """Calculate the shape of a matrix as a list of integers."""
    
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
