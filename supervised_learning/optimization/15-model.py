#!/usr/bin/env python3
"""Full training model with Adam, BN, LR decay, mini-batch"""

import numpy as np
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        Z, mean, var, beta, gamma, epsilon
    )
    return Z_norm


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]])

    global_step = tf.Variable(0, trainable=False)

    # learning rate decay (stepwise inverse time)
    alpha_decay = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=True
    )

    # Xavier init equivalent
    init = tf.contrib.layers.variance_scaling_initializer()

    prev = x

    # build network
    for i in range(len(layers)):
        W = tf.Variable(
            init([prev.shape[-1], layers[i]]),
            trainable=True
        )
        b = tf.Variable(tf.zeros([layers[i]]), trainable=True)

        Z = tf.matmul(prev, W) + b

        if i < len(layers) - 1:
            gamma = tf.Variable(tf.ones([layers[i]]), trainable=True)
            beta = tf.Variable(tf.zeros([layers[i]]), trainable=True)
            Z = batch_norm(Z, gamma, beta, epsilon)
            prev = activations[i](Z)
        else:
            prev = Z

    logits = prev

    loss = tf.losses.softmax_cross_entropy(y, logits)

    pred = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    m = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):

            X_s, Y_s = X_train.copy(), Y_train.copy()
            perm = np.random.permutation(m)
            X_s, Y_s = X_s[perm], Y_s[perm]

            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            step = 0

            for i in range(0, m, batch_size):
                step += 1

                X_batch = X_s[i:i + batch_size]
                Y_batch = Y_s[i:i + batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    c = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    a = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})

                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(c))
                    print("\t\tAccuracy: {}".format(a))

        save_path = saver.save(sess, save_path)

    return save_path
