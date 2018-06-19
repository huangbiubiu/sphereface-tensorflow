# -*- coding: utf-8 -*-
import tensorflow as tf


def softmax_loss(logits, labels):
    """
    Original softmax with cross entropy
    :param logits: output of network
    :param labels: onehot label
    :return: loss value
    """
    with tf.name_scope("softmax_with_cross_entropy"):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

        return tf.reduce_mean(loss)




pass
