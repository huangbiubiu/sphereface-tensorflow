# -*- coding: utf-8 -*-
import tensorflow as tf
import math

from tensorflow.python.framework import ops


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
                cos_th3 = tf.pow(cos_th, 3)
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

            ###
            # scale grad
            ###
            # normalize gradient

            coeff_x = sign3 * (-24 * cos_th4 + 8 * cos_th2 + 1) + sign4
            coeff_w = sign3 * (32 * cos_th3 - 16 * cos_th)
            coeff_norm = tf.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)

            @ops.RegisterGradient("margin_grad_norm")
            def grad_norm(unused_op, grad):
                diff_at_label = tf.gather_nd(grad, ordinal_y)  # extract label position derivative
                diff_norm = diff_at_label / coeff_norm
                return grad + tf.scatter_nd(ordinal_y, diff_norm - diff_at_label, tf.shape(grad, out_type=tf.int64))

            with tf.get_default_graph().gradient_override_map({"Identity": "margin_grad_norm"}):
                comb_logits_diff = tf.identity(comb_logits_diff, name="Identity")

            ###
            # scale grad completed
            ###

            updated_logits = ff * logits + f * comb_logits_diff

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))
        return logits, loss


# modified grad op version
# @tf.custom_gradient
def margin_inner_product_layer(x, label, num_output, m, base_, gamma_, power_, lambda_min_, iter_):
    with tf.name_scope("margin_inner_product_layer"):
        with tf.name_scope("lambda"):
            lambda_ = base_ * math.pow((1. + gamma_ * iter_), -power_)
            lambda_ = max(lambda_, lambda_min_)

        # all symbol in this implementation is same as original caffe implementation
        M_ = x.get_shape()[0]  # batch size
        K_ = x.get_shape()[1]  # input feature length
        N_ = num_output

        weight = tf.get_variable("W", shape=[N_, K_], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        norm_w = tf.norm(weight, axis=1, name='norm_w')  # (N, )
        weight = tf.div(weight, tf.reshape(norm_w, (-1, 1)))

        norm_x = tf.norm(x, axis=1, name='norm_x')  # (M,)
        wx = tf.matmul(x, tf.transpose(weight))  # x: (M, K), w: (N, K), wx: (M, N)

        cos_theta = wx / tf.reshape(norm_x, shape=[-1, 1])

        # indices of label
        ordinal = tf.range(0, tf.shape(x, out_type=tf.int64)[0], dtype=tf.int64)
        ordinal_y = tf.stack([ordinal, label], axis=1)

        if m == 4:
            cos_theta_y = tf.gather_nd(cos_theta, ordinal_y)  # (M, 1)
            cos_theta_quadratic = tf.pow(cos_theta_y, 2, 'cos_theta_quadratic')
            cos_theta_cubic = tf.pow(cos_theta_y, 3, 'cos_theta_cubic')
            cos_theta_quartic = tf.pow(cos_theta_y, 4, 'cos_theta_quartic')

            sign_0 = tf.sign(cos_theta_y)
            # sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
            sign_3 = tf.multiply(sign_0, tf.sign(2 * cos_theta_quadratic - 1))  # (-1)^k
            # sign_4 = 2 * sign_0 + sign_3 - 3
            sign_4 = 2 * sign_0 + sign_3 - 3  # -2k

            # forward pass
            phi_theta = norm_x * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
            phi_theta -= norm_x * cos_theta_y
            phi_theta = tf.scatter_nd(ordinal_y, phi_theta, tf.shape(cos_theta, out_type=tf.int64))  # (M, N)
            phi_theta += wx

            logits = tf.add(phi_theta, lambda_ * wx) / (1 + lambda_)

            return logits
            pass
        else:
            raise NotImplementedError(f"m={m} is not implemented.")

    pass
