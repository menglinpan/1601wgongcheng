from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

LOGDIR = "temp/mnist/demo/3/"
def conv_layer(input, size_in, size_out, name="conv"):
    # define the convolution layer
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def fc_layer(input, size_in, size_out, name="fc"):
    # define the fully-connected layer
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", act)
        return act

def mnist():
    sess = tf.Session()
    # Setup placeholders, and reshape the data
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        tf.summary.image("image", x_image, max_outputs=3)
    conv1 = conv_layer(x_image, 1, 32, name="conv1")
    conv1_pool = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(conv1_pool, 32, 64, name="conv2")
    conv_out = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    fc1 = tf.nn.relu(fc_layer(flattened, 7 * 7 * 64, 1024, name="fc1"))

    logits = fc_layer(fc1, 1024, 10, name="logits")

    with tf.name_scope("cross_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        tf.summary.scalar("loss", xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    mnist = input_data.read_data_sets(train_dir="temp/mnist/data", one_hot=True)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    file_writer = tf.summary.FileWriter(LOGDIR)
    file_writer.add_graph(sess.graph)
    merge_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [summaries, _] = sess.run([merge_op, accuracy], feed_dict={x: batch[0], y: batch[1]})
            file_writer.add_summary(summaries, i)
        if i % 400 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print("step: %d accuracy: %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

if __name__ == "__main__":
    mnist()







