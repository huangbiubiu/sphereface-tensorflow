# -*- coding: utf-8 -*-
import model.layers
from model.NerualNetwork import NerualNetwork

import tensorflow as tf
import tflearn

from model.loss import softmax_loss


class NaiveCNN(NerualNetwork):
    def inference(self, images, num_class, label, param):
        tf.summary.image("input_image", images)

        features = self.Network(images)

        if param['softmax'] == 'vanilla':
            logits = tf.layers.dense(features, num_class, name="output")
            loss = softmax_loss(logits, label)
        elif param['softmax'] == 'a-softmax':
            logits, loss = model.layers.Loss_ASoftmax(features, tf.argmax(label, axis=1), 1.0, num_class, m=2)
        else:
            raise ValueError(f"Softmax {param['softmax']} is not supported.")
        tf.summary.histogram("output", logits)

        return logits, loss

    def Network(self, data_input, training=True):
        x = tflearn.conv_2d(data_input, 32, 3, strides=1, activation='prelu', weights_init='xavier')
        x = tflearn.conv_2d(x, 32, 3, strides=2, activation='prelu', weights_init='xavier')
        x = tflearn.conv_2d(x, 64, 3, strides=1, activation='prelu', weights_init='xavier')
        x = tflearn.conv_2d(x, 64, 3, strides=2, activation='prelu', weights_init='xavier')
        x = tflearn.conv_2d(x, 128, 3, strides=1, activation='prelu', weights_init='xavier')
        x = tflearn.conv_2d(x, 128, 3, strides=2, activation='prelu', weights_init='xavier')
        x = tflearn.flatten(x)

        feat = tflearn.fully_connected(x, 2, weights_init='xavier')

        return feat
