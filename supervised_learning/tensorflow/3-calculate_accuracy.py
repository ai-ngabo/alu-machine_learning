#!/usr/bin/env python3
"""TensorFlow accuracy calculation module."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.Tensor): Placeholder for the labels of the input data (one-hot
                       encoded).
        y_pred (tf.Tensor): Tensor containing the network's predictions.

    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the prediction.
    """
    # Get the index of the predicted class (highest value in y_pred)
    predicted_classes = tf.argmax(y_pred, axis=1)
    
    # Get the actual class index from one-hot encoded labels
    actual_classes = tf.argmax(y, axis=1)
    
    # Compare predictions with actual values
    correct_predictions = tf.equal(predicted_classes, actual_classes)
    
    # Convert boolean to float and calculate mean (accuracy)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy
