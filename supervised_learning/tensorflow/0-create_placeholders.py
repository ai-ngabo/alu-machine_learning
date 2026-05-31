#!/usr/bin/env python3
"""TensorFlow placeholders creation module."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for the input data and one-hot labels.

    Args:
        nx (int): The number of feature columns in the data.
        classes (int): The number of classes in the classifier.

    Returns:
        tuple: A tuple containing (x, y) where:
            - x (tf.placeholder): Placeholder for input data 
            - y (tf.placeholder): Placeholder for one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
