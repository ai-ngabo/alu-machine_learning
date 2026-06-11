#!/usr/bin/env python3
"""
Neural network model with:
- Mini-batch gradient descent
- Adam optimization
- Learning rate decay
- Batch normalization
- TensorFlow implementation (no external helpers)
"""

import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains and saves a neural network model using TensorFlow.
    """

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    A = x

    for i in range(len(layers)):
        init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

        Z = tf.layers.Dense(
            units=layers[i],
            kernel_initializer=init
        )(A)

        if i != len(layers) - 1:
            gamma = tf.Variable(tf.ones([1, layers[i]]), trainable=True)
            beta = tf.Variable(tf.zeros([1, layers[i]]), trainable=True)

            mean, variance = tf.nn.moments(Z, axes=[0])
            Z_norm = (Z - mean) / tf.sqrt(variance + epsilon)
            Z_tilde = gamma * Z_norm + beta

            A = activations[i](Z_tilde)
        else:
            A = Z

    tf.add_to_collection('y_pred', A)

    loss = tf.losses.softmax_cross_entropy(y, A)
    tf.add_to_collection('loss', loss)

    correct = tf.equal(tf.argmax(A, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    global_step = tf.Variable(0, trainable=False)

    lr = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        def run_eval(X, Y):
            return sess.run(
                (loss, accuracy),
                feed_dict={x: X, y: Y}
            )

        for epoch in range(epochs + 1):

            train_cost, train_acc = run_eval(X_train, Y_train)
            valid_cost, valid_acc = run_eval(X_valid, Y_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            X_shuff = np.random.permutation(X_train.shape[0])
            X_train_s = X_train[X_shuff]
            Y_train_s = Y_train[X_shuff]

            step = 0

            for i in range(0, X_train.shape[0], batch_size):
                step += 1

                X_batch = X_train_s[i:i + batch_size]
                Y_batch = Y_train_s[i:i + batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    c, a = run_eval(X_batch, Y_batch)
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(c))
                    print("\t\tAccuracy {}".format(a))

        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
