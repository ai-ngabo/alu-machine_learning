#!/usr/bin/env python3
"""Builds, trains, and saves a neural network model in TensorFlow"""

import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates placeholders for input data and labels"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    W = tf.Variable(init([prev.shape.as_list()[-1], n]), name='W')
    b = tf.Variable(tf.zeros([n]), name='b')
    Z = tf.matmul(prev, W) + b

    if activation is None:
        return Z

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')
    epsilon = 1e-8
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    return activation(Z_norm)


def forward_prop(x, layers, activations):
    """Creates the forward propagation graph with batch normalization"""
    A = x
    for i in range(len(layers)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss"""
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    )


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the Adam optimizer"""
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates inverse time decay learning rate operation"""
    return tf.train.inverse_time_decay(
        alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow.

    Returns: the path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    m = X_train.shape[0]

    # Build graph
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Learning rate with decay
    alpha_decayed = learning_rate_decay(alpha, decay_rate, global_step, 1)

    # Adam optimizer using decayed learning rate
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha_decayed,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss, global_step=global_step)

    # Number of complete batches + possible remainder
    num_batches = int(np.ceil(m / batch_size))

    # Save collections for later loading
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', optimizer)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            # Print epoch metrics (before training starts and after each epoch)
            train_cost, train_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_acc = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            # Shuffle training data before each epoch
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            # Mini-batch gradient descent
            for step in range(num_batches):
                start = step * batch_size
                end = min(start + batch_size, m)

                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch})

                # Print every 100 steps (1-indexed)
                if (step + 1) % 100 == 0:
                    step_cost, step_acc = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch}
                    )
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy {}".format(step_acc))

        # Save the model
        saved_path = saver.save(sess, save_path)

    return saved_path
