# -*- coding: utf-8 -*-
import tensorflow as tf


# TODO test the implementation
def a_softmax(features, class_num, m, global_steps, base=1000, gamma=0.12, power=1, lambda_min=5):
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
        cos_theta = tf.div(xw, tf.reshape(x_norm, (-1, 1)))

        margin_cos_cal = {2: double_margin, 3: triple_margin, 4: quadruple_margin}
        cos_m_theta = margin_cos_cal[m](cos_theta)

        logits = (cos_m_theta + lba * xw) / (1 + lba)

        # (|W||x|cos(m*theta) + labmda*W*x) / (1+lambda), see 1612.02295, sec 5.1
        # logits = tf.div(
        #     (tf.matmul(x_norm, w_norm, name='xw_norm_multiply') * tf.cos(theta * m, name='cos_m_theta') + lba * xw),
        #     1 + lba, name='logits')
        tf.summary.scalar("a-softmax_lambda", lba)

        # logits = tf.Print(logits, [theta], "logits=", summarize=100, name='print_logits')
        return logits


def double_margin(cos_theta):
    with tf.name_scope('double_margin'):
        cos_sign = tf.sign(cos_theta)
        return 2 * tf.multiply(cos_sign, tf.square(cos_theta)) - 1
    pass


def triple_margin(cos_theta):
    raise NotImplementedError("m = 3 is not implemented")


def quadruple_margin(cos_theta):
    with tf.name_scope("quadruple_margin"):
        cos_th2 = tf.square(cos_theta)
        cos_th4 = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
        sign4 = 2 * sign0 + sign3 - 3
        res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4

    return res


def Loss_ASoftmax(x, y, l, num_cls, m=2, name='asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda
    '''
    with tf.name_scope("Loss_ASoftmax"):
        xs = x.get_shape()
        w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())  # shape(D, C)

        eps = 1e-8

        xw = tf.matmul(x, w)  # shape(B,C)

        if m == 0:
            return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

        w_norm = tf.norm(w, axis=0) + eps  # shape(1, C)
        logits = xw / w_norm  # shape(B, C)

        if y is None:
            return logits, None

        # ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
        ordinal = tf.range(0, tf.shape(x, out_type=tf.int64)[0], dtype=tf.int64)
        ordinal_y = tf.stack([ordinal, y], axis=1)

        x_norm = tf.norm(x, axis=1) + eps  # shape (B, 1)

        sel_logits = tf.gather_nd(logits, ordinal_y)  # z_j, shape (B, 1)

        cos_th = tf.div(sel_logits, x_norm)  # shape (B, 1) cos(theta) for each sample to corresponding correct class

        if m == 1:

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                  (labels=y, logits=logits))

        else:

            if m == 2:
                cos_sign = tf.sign(cos_th)
                res = 2 * tf.multiply(cos_sign, tf.square(cos_th)) - 1
            elif m == 4:
                cos_th2 = tf.square(cos_th)
                cos_th4 = tf.pow(cos_th, 4)
                sign0 = tf.sign(cos_th)
                sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
                sign4 = 2 * sign0 + sign3 - 3
                res = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
            else:
                raise ValueError('unsupported value of m')

            scaled_logits = tf.multiply(res, x_norm)  # shape (B, 1): |x|psi()

            f = 1.0 / (1.0 + l)
            ff = 1.0 - f
            comb_logits_diff = tf.add(logits,
                                      tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits),
                                                    tf.shape(logits, out_type=tf.int64)))
            updated_logits = ff * logits + f * comb_logits_diff

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))
        return logits, loss
