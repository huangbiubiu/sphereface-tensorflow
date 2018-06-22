# -*- coding: utf-8 -*-

import tensorflow as tf

import model.layers
from model.NerualNetwork import NerualNetwork


class SphereCNN(NerualNetwork):
    @staticmethod
    def __res_block(data_in,
                    kernel_size: int,
                    filters: int,
                    name: str,
                    strides=1,
                    conv_first=True,
                    activation_name='prelu'):
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
                                     name=name + '_1')
        else:
            conv1 = data_in
        conv2 = tf.layers.conv2d(conv1,
                                 filters=filters,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='SAME',
                                 activation=activation,
                                 name=name + '_2')
        conv3 = tf.layers.conv2d(conv2,
                                 filters=filters,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='SAME',
                                 activation=activation,
                                 name=name + '_3')
        res = tf.add(conv1, conv3, name=name + '_res')
        # tf.summary.image(name + '_res', res)
        return res
        pass

    def inference(self, images, num_class, param):
        # this implementation based on
        # https://github.com/wy1iu/sphereface/blob/master/train/code/sphereface_model.prototxt

        global_steps = param['global_steps']

        tf.summary.image("input_image", images)

        feature_map1 = self.__res_block(images, kernel_size=3, filters=64, strides=2, conv_first=True,
                                        name='conv1')
        feature_map2 = self.__res_block(feature_map1, kernel_size=3, filters=128, strides=2, conv_first=True,
                                        name='conv2')
        feature_map3 = self.__res_block(feature_map2, kernel_size=3, filters=256, name='conv3', conv_first=False)
        feature_map4 = self.__res_block(feature_map3, kernel_size=3, filters=256, name='conv4', conv_first=True,
                                        strides=2)
        feature_map5 = self.__res_block(feature_map4, kernel_size=3, filters=256, name='conv5', conv_first=False)
        feature_map6 = self.__res_block(feature_map5, kernel_size=3, filters=512, name='conv6', conv_first=False)
        feature_map7 = self.__res_block(feature_map6, kernel_size=3, filters=512, name='conv6', conv_first=False)
        feature_map8 = self.__res_block(feature_map7, kernel_size=3, filters=512, name='conv6', conv_first=True,
                                        strides=2)

        # features = tf.reshape(images, [images.get_shape().as_list()[0], -1], name="flatten")
        features = tf.layers.Flatten()(feature_map8)
        features = tf.layers.dense(features, 512, activation=None, name="fc5")

        # output layer
        # logits = tf.layers.dense(features, num_class, name="output")
        # logits = model.layers.a_softmax(features, num_class, m=3, global_steps=global_steps)
        if param['softmax'] == 'vanilla':
            logits = tf.layers.dense(features, num_class, name="output")
        elif param['softmax'] == 'a-softmax':
            logits = model.layers.a_softmax(features, num_class, m=3, global_steps=global_steps)
        else:
            raise ValueError(f"Softmax {param['softmax']} is not supported.")
        tf.summary.histogram("output", logits)

        return logits
