#!/usr/bin/env python3
"""Mini-batch training module"""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network using mini-batch gradient descent
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        graph = tf.get_default_graph()

        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]
        train_op = graph.get_collection('train_op')[0]

        m = X_train.shape[0]

        for epoch in range(epochs + 1):

            # shuffle data each epoch (except still OK at epoch 0 for evaluation)
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})

            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            # do not train after last epoch evaluation
            if epoch == epochs:
                break

            step = 0

            for i in range(0, m, batch_size):
                step += 1

                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})

                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path
