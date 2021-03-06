# -*- coding: utf-8 -*-

import tensorflow as tf

import model.layers
from model import GraphType
from model.NerualNetwork import NerualNetwork

# TODO remove bias as paper mentioned
from model.loss import softmax_loss


class SphereCNN(NerualNetwork):
    @staticmethod
    def __res_block(data_in,
                    kernel_size: int,
                    filters: int,
                    name: str,
                    strides=1,
                    conv_first=True,
                    activation_name='relu',
                    bias_regularizer=None,
                    weight_regularizer=None,
                    remove_last_activation=False):
        def prelu(_x, scope=None):
            """parametric ReLU activation"""
            # reference https://stackoverflow.com/a/44947501/5634636
            with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
                _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                         dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
                return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

        activation_name = activation_name.lower()
        if activation_name == 'prelu':
            activation = prelu
        elif activation_name == 'relu':
            activation = tf.nn.relu
        else:
            raise ValueError(f"Activation {activation_name} is not supported.")

        if conv_first:
            conv1 = tf.layers.conv2d(data_in,
                                     filters=filters,
                                     kernel_size=[kernel_size, kernel_size],
                                     padding='SAME',
                                     strides=(strides, strides),
                                     activation=activation,
                                     name=name + '_1',
                                     kernel_regularizer=weight_regularizer,
                                     bias_regularizer=bias_regularizer)
        else:
            conv1 = data_in
        conv2 = tf.layers.conv2d(conv1,
                                 filters=filters,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='SAME',
                                 activation=activation,
                                 name=name + '_2',
                                 kernel_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer)
        conv3 = tf.layers.conv2d(conv2,
                                 filters=filters,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='SAME',
                                 activation=None if remove_last_activation else activation,
                                 name=name + '_3',
                                 kernel_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer)
        res = tf.add(conv1, conv3, name=name + '_res')
        # tf.summary.image(name + '_res', res)
        return res
        pass

    def inference(self, images, num_class, label=None, param=None):
        # this implementation based on
        # https://github.com/wy1iu/sphereface/blob/master/train/code/sphereface_model.prototxt

        global_steps = param['global_steps']
        weight_regularizer = param['weight_regularizer'] if 'weight_regularizer' in param else None
        bias_regularizer = param['bias_regularizer'] if 'bias_regularizer' in param else None

        tf.summary.image("input_image", images)

        # image_size = self.get_image_size(param)
        # images = tf.image.resize_images(images, image_size)
        # images = images[:, :, :, :3]  # discard the alpha channel
        # images.set_shape((None, *image_size, 3))
        if 'image_size' in param:
            tf.logging.warn("param 'image_size' is deprecated")

        feature_map1 = self.__res_block(images, kernel_size=3, filters=64, strides=2, conv_first=True,
                                        name='conv1_1', weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map2 = self.__res_block(feature_map1, kernel_size=3, filters=128, strides=2, conv_first=True,
                                        name='conv2_1', weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map3 = self.__res_block(feature_map2, kernel_size=3, filters=128, name='conv2_2', conv_first=False,
                                        weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map4 = self.__res_block(feature_map3, kernel_size=3, filters=256, name='conv3_1', conv_first=True,
                                        strides=2, weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map5 = self.__res_block(feature_map4, kernel_size=3, filters=256, name='conv3_2', conv_first=False,
                                        weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map6 = self.__res_block(feature_map5, kernel_size=3, filters=256, name='conv3_3', conv_first=False,
                                        weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map7 = self.__res_block(feature_map6, kernel_size=3, filters=256, name='conv3_4', conv_first=False,
                                        weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer)
        feature_map8 = self.__res_block(feature_map7, kernel_size=3, filters=512, name='conv4', conv_first=True,
                                        strides=2, weight_regularizer=weight_regularizer,
                                        bias_regularizer=bias_regularizer, remove_last_activation=True)

        # features = tf.reshape(images, [images.get_shape().as_list()[0], -1], name="flatten")
        features = tf.layers.Flatten()(feature_map8)
        features = tf.layers.dense(features, 512, activation=None, name="fc5",
                                   kernel_regularizer=weight_regularizer,
                                   bias_regularizer=bias_regularizer)

        if 'graph_type' in param:
            graph_type = param['graph_type']
        else:
            graph_type = GraphType.TRAIN

        if graph_type != GraphType.TRAIN:
            return features

        # output layer
        # logits = tf.layers.dense(features, num_class, name="output")
        # logits = model.layers.a_softmax(features, num_class, m=3, global_steps=global_steps)
        if param['softmax'] == 'vanilla':
            logits = tf.layers.dense(features, num_class, name="output",
                                     kernel_regularizer=weight_regularizer,
                                     bias_regularizer=bias_regularizer)
            return logits, softmax_loss(logits, label)
        elif param['softmax'] == 'a-softmax':
            if 'base_lambda' in param:
                lambda_base = param['base_lambda']
            else:
                lambda_base = 1000  # default number as paper
            gamma = 0.12
            power = 1
            lambda_min = 5
            lba = tf.cast(
                tf.maximum(lambda_base * tf.pow(1 + gamma * tf.cast(global_steps, tf.float64), -power), lambda_min),
                tf.float32, name='calculate_lambda')
            tf.summary.scalar("lambda", lba)
            # logits, loss = model.layers.Loss_ASoftmax(features, tf.argmax(label, axis=1), lba, num_class, m=4)
            margin_logits, logits = model.layers.margin_inner_product_layer(features,
                                                                            tf.argmax(label, axis=1),
                                                                            num_class,
                                                                            global_steps)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.argmax(label, axis=1),
                                                              logits=margin_logits)
            loss = tf.reduce_mean(loss)
        else:
            raise ValueError(f"Softmax {param['softmax']} is not supported.")
        tf.summary.histogram("output", logits)

        return logits, loss
