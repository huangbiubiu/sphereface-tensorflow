# -*- coding: utf-8 -*-
import model.layers
from model.NerualNetwork import NerualNetwork

import tensorflow as tf


class NaiveCNN(NerualNetwork):
    def inference(self, images, num_class, param):
        tf.summary.image("input_image", images)

        images = tf.layers.conv2d(images,
                                  filters=32,
                                  kernel_size=[5, 5],
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='conv1')
        images = tf.layers.max_pooling2d(images, [3, 3], [1, 1], name='pool1')
        images = tf.layers.batch_normalization(images, name='norm1')

        images = tf.layers.conv2d(images,
                                  filters=32,
                                  kernel_size=[3, 3], activation=tf.nn.relu,
                                  padding='same', name='conv2')
        images = tf.layers.max_pooling2d(images, [2, 2], [1, 1], name='pool2')
        images = tf.layers.batch_normalization(images, name='norm2')

        # features = tf.reshape(images, [images.get_shape().as_list()[0], -1], name="flatten")
        features = tf.layers.Flatten()(images)
        features = tf.layers.dense(features, 256, activation=tf.nn.relu, name="local1")

        if param['softmax'] == 'vanilla':
            logits = tf.layers.dense(features, num_class, name="output")
        elif param['softmax'] == 'a-softmax':
            logits = model.layers.a_softmax(features, num_class, m=3, global_steps=param['global_steps'])
        else:
            raise ValueError(f"Softmax {param['softmax']} is not supported.")
        tf.summary.histogram("output", logits)

        return logits
