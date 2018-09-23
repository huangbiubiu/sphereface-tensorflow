# -*- coding: utf-8 -*-

# import tensorflow as tf
# from util.custom_gradient import custom_gradient
#
#
# @custom_gradient
# def margin_inner_product_layer(x: tf.Tensor, label, iter_, num_output: int, m=4, base_=1000, gamma_=0.12, power_=-1,
#                                lambda_min_=5):
#     with tf.name_scope("margin_inner_product_layer"):
#         with tf.name_scope("lambda"):
#             lambda_ = base_ * tf.pow((1. + gamma_ * tf.cast(iter_, tf.float32)), -power_)
#             lambda_ = tf.maximum(lambda_, lambda_min_)
#         # all symbol in this implementation is same as original caffe implementation
#         M_ = tf.shape(x)[0]  # batch size
#         K_ = x.shape[1]  # input feature length
#         N_ = num_output  # TODO num_output is a Tensor because the convert by decorator @tf.custom_gradient
#         weight = tf.get_variable("W", shape=[N_, K_], dtype=tf.float32,
#                                  use_resource=True,
#                                  initializer=tf.contrib.layers.xavier_initializer())
#         norm_w = tf.norm(weight, axis=1, name='norm_w')  # (N, )
#         weight = tf.div(weight, tf.reshape(norm_w, (-1, 1)))
#         norm_x = tf.norm(x, axis=1, name='norm_x')  # (M,)
#         wx = tf.matmul(x, tf.transpose(weight))  # x: (M, K), w: (N, K), wx: (M, N)
#         cos_theta = wx / tf.reshape(norm_x, shape=[-1, 1])
#         # indices of label
#         ordinal = tf.range(0, tf.shape(x, out_type=tf.int64)[0], dtype=tf.int64)
#         ordinal_y = tf.stack([ordinal, label], axis=1)
#         if m == 4:
#             cos_theta_y = tf.gather_nd(cos_theta, ordinal_y)  # (M, 1)
#             cos_theta_quadratic = tf.pow(cos_theta_y, 2, 'cos_theta_quadratic')
#             cos_theta_cubic = tf.pow(cos_theta_y, 3, 'cos_theta_cubic')
#             cos_theta_quartic = tf.pow(cos_theta_y, 4, 'cos_theta_quartic')
#             sign_0 = tf.sign(cos_theta_y)
#             # sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
#             sign_3 = tf.multiply(sign_0, tf.sign(2 * cos_theta_quadratic - 1))  # (-1)^k
#             # sign_4 = 2 * sign_0 + sign_3 - 3
#             sign_4 = 2 * sign_0 + sign_3 - 3  # -2k
#             # forward pass
#             phi_theta = norm_x * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
#             phi_theta -= norm_x * cos_theta_y
#             phi_theta = tf.scatter_nd(ordinal_y, phi_theta, tf.shape(cos_theta, out_type=tf.int64))  # (M, N)
#             phi_theta += wx
#             logits = tf.add(phi_theta, lambda_ * wx) / (1 + lambda_)
#             pass
#         else:
#             raise NotImplementedError(f"m={m} is not implemented.")
#         pass
#
#     def back_prop(dx, dwx, variables=None):
#
#         coeff_x = sign_3 * (-24 * cos_theta_quartic + 8 * cos_theta_quadratic + 1) + sign_4
#         coeff_w = sign_3 * (32 * cos_theta_cubic - 16 * cos_theta_y)
#         coeff_norm = tf.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)
#
#         dw = tf.gradients(logits, weight)[0]
#
#         def normalize_derivative(diff):
#             diff_at_label = tf.gather_nd(diff, ordinal_y)  # extract label position derivative
#             diff_norm = diff_at_label / coeff_norm
#             return diff + tf.scatter_nd(ordinal_y, diff_norm - diff_at_label, tf.shape(diff, out_type=tf.int64))
#
#         # return [normalize_derivative(dx), normalize_derivative(dw), grad_ys[2]]
#         return [normalize_derivative(dx), dwx], [normalize_derivative(dw)]
#         pass
#
#     return [logits, wx], back_prop

import tensorflow as tf
from tensorflow.python.framework import ops


def margin_inner_product_layer(x: tf.Tensor, label, iter_, num_output: int, m=4, base_=1000, gamma_=0.12, power_=-1,
                               lambda_min_=5):
    with tf.name_scope("margin_inner_product_layer"):
        with tf.name_scope("lambda"):
            lambda_ = base_ * tf.pow((1. + gamma_ * tf.cast(iter_, tf.float32)), -power_)
            lambda_ = tf.maximum(lambda_, lambda_min_)
        # all symbol in this implementation is same as original caffe implementation
        M_ = tf.shape(x)[0]  # batch size
        K_ = x.shape[1]  # input feature length
        N_ = num_output
        weight = tf.get_variable("W", shape=[N_, K_], dtype=tf.float32,
                                 use_resource=True,
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
            phi_theta += wx  # logits with margin

            # normalize gradient

            coeff_x = sign_3 * (-24 * cos_theta_quartic + 8 * cos_theta_quadratic + 1) + sign_4
            coeff_w = sign_3 * (32 * cos_theta_cubic - 16 * cos_theta_y)
            coeff_norm = tf.sqrt(coeff_w * coeff_w + coeff_x * coeff_x)

            @ops.RegisterGradient("margin_grad_norm")
            def grad_norm(unused_op, grad):
                diff_at_label = tf.gather_nd(grad, ordinal_y)  # extract label position derivative
                diff_norm = diff_at_label / coeff_norm
                return grad + tf.scatter_nd(ordinal_y, diff_norm - diff_at_label, tf.shape(grad, out_type=tf.int64))

            with tf.get_default_graph().gradient_override_map({"Identity": "margin_grad_norm"}):
                phi_theta_grad = tf.identity(phi_theta, name="Identity")

            # balance margin function and original softmax with parameter lambda
            logits = tf.add(phi_theta_grad, lambda_ * wx) / (1 + lambda_)

            return logits, wx

        else:
            raise NotImplementedError(f"m={m} is not implemented.")
        pass
