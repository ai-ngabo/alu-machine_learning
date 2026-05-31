#!/usr/bin/env python3
"""TensorFlow training module."""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training labels (one-hot encoded).
        X_valid (np.ndarray): Validation input data.
        Y_valid (np.ndarray): Validation labels (one-hot encoded).
        layer_sizes (list): Number of nodes in each layer of the network.
        activations (list): Activation functions for each layer.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train over.
        save_path (str): Path where to save the model.

    Returns:
        str: The path where the model was saved.
    """
    # Get the number of features and classes
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    # Create placeholders
    x, y = create_placeholders(nx, classes)
    # Build forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    # Create training operation
    train_op = create_train_op(loss, alpha)
    # Create saver
    saver = tf.train.Saver()
    # Initialize variables
    init = tf.global_variables_initializer()
    # Start session
    with tf.Session() as sess:
        sess.run(init)
        # Training loop
        for i in range(iterations + 1):
            # Run training operation for all iterations except the first
            if i > 0:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            # Calculate metrics every 100 iterations, at iteration 0,
            # and at the final iteration
            if i % 100 == 0 or i == iterations:
                # Training metrics
                train_cost, train_accuracy = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_train, y: Y_train}
                )
                # Validation metrics
                valid_cost, valid_accuracy = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_valid, y: Y_valid}
                )
                # Print results with proper formatting (tabs and newlines)
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))
        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
