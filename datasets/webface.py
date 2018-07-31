# -*- coding: utf-8 -*-
import csv
import os
import pickle
import random

import cv2
import numpy as np
import tensorflow as tf

import tools.cp2tform_np
from datasets.Dataset import Dataset


class WebFace(Dataset):

    def load_data(self, data_dir, is_training, epoch_num, batch_size, data_param):
        data_dir = os.path.expanduser(data_dir)

        # construct dataset with python code
        face_classes = sorted(os.listdir(data_dir))
        num_class = len(face_classes)
        # TODO this way to build label may raise a bug, see:
        # https://stackoverflow.com/questions/43646266/tensorflow-loss-jumps-up-after-restoring-rnn-net
        tf.logging.info('loading dataset from disk')
        class2index = {k: v for v, k in enumerate(face_classes)}
        data_list = map(lambda cls: list_images(data_dir, cls, class2index), face_classes)
        # flat the list
        data_list = [item for sublist in data_list for item in sublist]
        random.shuffle(data_list) # the loss will not converge if not shuffle

        tf.logging.info('dataset loaded')

        # with open('/home/hyh/tmp/dlist.pkl', 'rb') as f:
        #     data_list = pickle.load(f)
        # num_class = len(set(map(lambda x: x[1], data_list)))

        # construct tensorflow dataset graph
        with tf.name_scope("load_webface"):
            def decode_data(image_path: str, label):
                """
                decode path to tensor
                :param image_path: path of image to be decoded
                :param label: training label of image
                :return: augmented image and corresponding one hot label.
                augmented image will be all zero when failed to detect faces in image
                """
                # read image from file
                image_file = tf.read_file(image_path)
                image_decoded = tf.image.decode_image(image_file)
                # convert gray scale image to RGB
                image_decoded = tf.cond(tf.equal(tf.shape(image_decoded)[2], 1),
                                        lambda: tf.image.grayscale_to_rgb(image_decoded),
                                        lambda: image_decoded)

                # crop image
                with tf.name_scope("image_alignment"):
                    # read preprocessed image directly from disk
                    image_transformed = image_decoded
                    image_transformed.set_shape([112, 96, 3])
                    # image_transformed.set_shape([32, 32, 3]) ONLY FOR DEBUG

                with tf.name_scope("image_normalization"):
                    # the implementation of normalization in 1704.08063 Sec 4.1
                    image_normalized = tf.div(tf.subtract(tf.cast(image_transformed, tf.float32), 127.5), 128)
                with tf.name_scope("data_augmentation"):
                    image_augmented = tf.image.random_flip_left_right(image_normalized)

                return image_augmented, tf.one_hot(label, depth=num_class)
                # return tf.cast(image_transformed, tf.float32), tf.one_hot(label, depth=num_class) FOR DEBUG

            dataset = tf.data.Dataset.from_tensor_slices(
                (list(map(lambda item: item[0], data_list)),
                 list(map(lambda item: item[1], data_list))))

            dataset = dataset.prefetch(batch_size * 10)
            dataset = dataset.map(decode_data)

            if is_training:
                dataset = dataset.shuffle(10 * batch_size, seed=666).repeat(epoch_num).batch(batch_size)
                # dataset = dataset.repeat(epoch_num).batch(batch_size)
                pass
            else:
                dataset = dataset.batch(batch_size)

            return dataset.make_one_shot_iterator().get_next(), num_class


def list_images(data_dir, cls_name, cls2idx):
    cls_path = os.path.join(data_dir, cls_name)
    images = os.listdir(cls_path)

    if cls2idx is not None:
        images_data = list(
            map(lambda img_file_name: (os.path.join(cls_path, img_file_name), cls2idx[cls_name]), images))
    else:
        raise ValueError("None parameter cls2idx is not supported")
        # images_data = list(
        #     map(lambda img_file_name: (os.path.join(cls_path, img_file_name), cls_name), images))
    return images_data

