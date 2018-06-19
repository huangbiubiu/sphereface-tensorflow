# -*- coding: utf-8 -*-
import tensorflow as tf


def a_softmax(features, class_num, lba, m):
    """
    Implementation of A-Softmax with out cross entropy.
    See arXiv: 1704.08063.
    :param features: flatten input features in (B, D), where B is batch size
    :param class_num: class number in classification
    :param lba: lambda
    :param m: m in theta
    :return: output logits
    """
    # based on https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/Loss_ASoftmax.py
    with tf.name_scope("l-softmax"):
        x = features  # shape (B, D)
        w = tf.get_variable(name='softmax_loss/w',
                            shape=[features.get_shape()[1], class_num],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())  # shape (D, num_cls)

        eps = 1e-8  # prevent w from zero, TODO try zero eps
        w_norm = tf.norm(w, axis=0) + eps
        x_norm = tf.norm(x, axis=1) + eps

        xw = tf.matmul(x, w)  # in shape (B, num_cls)
        theta = tf.acos(tf.div(tf.multiply(w_norm, x_norm), xw))  # in shape (B, num_cls)

        # (|W||x|cos(m*theta) + labmda*W*x) / (1+lambda), see 1612.02295, sec 5.1
        logits = (w_norm * x_norm * tf.cos(theta * m) + lba * xw) / (1 + lba)

        return logits
