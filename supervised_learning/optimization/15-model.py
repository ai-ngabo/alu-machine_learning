#!/usr/bin/env python3
"""Full neural network training pipeline with TF1"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]])

    # learning rate with stepwise inverse time decay
    global_step = tf.Variable(0, trainable=False)

    alpha_decay = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=True
    )

    # network construction
    prev = x

    for i in range(len(layers)):
        if i == len(layers) - 1:
            prev = tf.layers.Dense(
                units=layers[i],
                activation=None,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            )(prev)
        else:
            prev = create_batch_norm_layer(prev, layers[i], activations[i])

    logits = prev

    # loss + accuracy
    loss = tf.losses.softmax_cross_entropy(y, logits)
    y_pred = tf.nn.softmax(logits)

    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    train_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    m = X_train.shape[0]

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

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

                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]

                feed = {x: X_batch, y: Y_batch}

                sess.run(train_op, feed_dict=feed)

                if step % 100 == 0:
                    step_cost = sess.run(loss, feed_dict=feed)
                    step_acc = sess.run(accuracy, feed_dict=feed)

                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        save_path = saver.save(sess, save_path)

    return save_path
