#!/usr/bin/env python3
"""Mini-batch training"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a model using mini-batch gradient descent"""

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

            X_s, Y_s = shuffle_data(X_train, Y_train)

            feed_all_train = {x: X_train, y: Y_train}
            feed_all_valid = {x: X_valid, y: Y_valid}

            train_cost = sess.run(loss, feed_dict=feed_all_train)
            train_acc = sess.run(accuracy, feed_dict=feed_all_train)

            valid_cost = sess.run(loss, feed_dict=feed_all_valid)
            valid_acc = sess.run(accuracy, feed_dict=feed_all_valid)

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

                feed_batch = {x: X_batch, y: Y_batch}

                sess.run(train_op, feed_dict=feed_batch)

                if step % 100 == 0:
                    step_cost = sess.run(loss, feed_dict=feed_batch)
                    step_acc = sess.run(accuracy, feed_dict=feed_batch)

                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        save_path = saver.save(sess, save_path)

    return save_path
