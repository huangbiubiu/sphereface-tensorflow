# -*- coding: utf-8 -*-
import tensorflow as tf

# TODO test the implementation
def a_softmax(features, class_num, m, global_steps, base=10000, gamma=0.12, power=1, lambda_min=5):
    """
    Implementation of A-Softmax with out cross entropy.
    See arXiv: 1704.08063.
    :param global_steps: steps for training
    :param base: base value of lambda
    :param gamma: gamma to calculate lambda
    :param power: power to calculate lambda
    :param lambda_min: minimum value of lambda
    :param features: flatten input features in (B, D), where B is batch size
    :param class_num: class number in classification
    :param lba: lambda
    :param m: m in theta
    :return: output logits, with shape (B, num_cls)
    """
    # the whole implementation reference
    # https://github.com/pppoe/tensorflow-sphereface-asoftmax/blob/master/Loss_ASoftmax.py

    with tf.name_scope("l-softmax"):
        # the lambda reference
        # https://github.com/wy1iu/sphereface/blob/master/tools/caffe-sphereface/src/caffe/layers/margin_inner_product_layer.cpp#L116
        lba = tf.cast(tf.maximum(base * tf.pow(1 + gamma * tf.cast(global_steps, tf.float64), -power), lambda_min),
                      tf.float32, name='calculate_lambda')

        x = features  # shape (B, D)
        w = tf.get_variable(name='softmax_loss/w',
                            shape=[features.get_shape()[1], class_num],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())  # shape (D, num_cls)

        eps = 1e-8  # prevent w from zero, TODO try zero eps
        w_norm = tf.reshape(tf.add(tf.norm(w, axis=0), eps), (1, class_num), name='normalize_w')  # shape (, num_cls)
        x_norm = tf.reshape(tf.add(tf.norm(x, axis=1), eps), (-1, 1), name='normalize_x')  # shape (B,)

        xw = tf.matmul(x, w, name='xw_matmul')  # in shape (B, num_cls)
        theta = tf.acos(tf.div(xw, tf.multiply(w_norm, x_norm, name='xw_norm')),
                        name='cal_theta')  # in shape (B, num_cls)

        # (|W||x|cos(m*theta) + labmda*W*x) / (1+lambda), see 1612.02295, sec 5.1
        logits = tf.div(
            (tf.matmul(x_norm, w_norm, name='xw_norm_multiply') * tf.cos(theta * m, name='cos_m_theta') + lba * xw),
            1 + lba, name='logits')
        tf.summary.scalar("a-softmax_lambda", lba)

        # logits = tf.Print(logits, [theta], "logits=", summarize=100, name='print_logits')
        return logits
