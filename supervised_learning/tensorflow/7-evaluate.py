#!/usr/bin/env python3
"""TensorFlow evaluation module."""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X (np.ndarray): Input data to evaluate.
        Y (np.ndarray): One-hot labels for X.
        save_path (str): Location to load the model from.

    Returns:
        tuple: (network prediction, accuracy, loss)
    """
    # Import the meta graph
    with tf.Session() as sess:
        # Load the meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')
        # Restore the model
        saver.restore(sess, save_path)
        # Get the graph
        graph = tf.get_default_graph()
        # Get tensors from the graph's collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        # Run evaluation
        y_pred_val, loss_val, accuracy_val = sess.run(
            [y_pred, loss, accuracy],
            feed_dict={x: X, y: Y}
        )
    return y_pred_val, accuracy_val, loss_val
