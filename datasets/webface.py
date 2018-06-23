# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class WebFace:

    @staticmethod
    def load_data(data_dir, is_training, epoch_num, batch_size):
        def list_images(cls_name, cls2idx):
            cls_path = os.path.join(data_dir, cls_name)
            images = os.listdir(cls_path)
            images_data = list(
                map(lambda img_file_name: (os.path.join(cls_path, img_file_name), cls2idx[cls_name]), images))

            return images_data

        # construct dataset with python code
        face_classes = os.listdir(data_dir)
        class2index = {k: v for v, k in enumerate(face_classes)}
        data_list = list(map(lambda cls: list_images(cls, class2index), face_classes))
        # flat the list
        data_list = [item for sublist in data_list for item in sublist]

        # construct tensorflow dataset graph
        with tf.name_scope("load_webface"):
            def decode_data(value):
                image_path = value[0]
                label = value[1]

                # read image from file
                image_file = tf.read_file(image_path)
                image_decoded = tf.image.decode_image(image_file)

                with tf.name_scope("image_normalization"):
                    # the implementation of 1704.08063 Sec 4.1
                    image_normalized = tf.div(tf.subtract(image_decoded, 127.5), 128)
                with tf.name_scope("data_augmentation"):
                    image_augmented = tf.image.random_flip_left_right(image_normalized)

                    return image_augmented, tf.one_hot(label)

            dataset = tf.data.Dataset.from_tensor_slices(data_list).prefetch(batch_size).map(decode_data)

            if is_training:

                dataset = dataset.shuffle(10 * batch_size).repeat(epoch_num).batch(batch_size)
            else:
                dataset = dataset.shuffle(10 * batch_size).batch(batch_size)

            return dataset.make_one_shot_iterator().get_next()

        pass

