# -*- coding: utf-8 -*-
import csv
import cv2
import os
import tensorflow as tf
import tensorflow.contrib as contrib
import tools.cp2tform_np
import numpy as np

from datasets.Dataset import Dataset
from tools.cp2tform import get_similarity_transform_for_cv2

import align.mtcnn.Aligner


class WebFace(Dataset):

    def load_data(self, data_dir, is_training, epoch_num, batch_size, data_param):
        data_dir = os.path.expanduser(data_dir)

        if 'image_size' in data_param:
            image_size = data_param['image_size']
            if isinstance(image_size, int):
                image_size = [image_size, image_size]
        else:
            image_size = [112, 96]

        # construct dataset with python code
        face_classes = os.listdir(data_dir)
        num_class = len(face_classes)
        class2index = {k: v for v, k in enumerate(face_classes)}
        data_list = map(lambda cls: list_images(data_dir, cls, class2index), face_classes)
        # flat the list
        data_list = [item for sublist in data_list for item in sublist]

        # fail_detect_images = get_fail_detect_files(data_param['fail_path'])
        # bounding_boxes = get_bounding_boxes(data_param['bounding_boxes'], margin=data_param['margin'])
        #
        # # remove images failed to detect faces
        # data_list = filter(lambda item: item[0] not in fail_detect_images, data_list)
        # # add bounding boxes to dataset
        # data_list = list(map(lambda item: [item[0], item[1], bounding_boxes[item[0]]], data_list))

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
                image_decoded = tf.cond(tf.equal(tf.shape(image_decoded)[2], 1),
                                        lambda: tf.image.grayscale_to_rgb(image_decoded),
                                        lambda: image_decoded)

                # aligner = align.mtcnn.Aligner.Aligner(image_size)

                # crop image
                with tf.name_scope("image_alignment"):
                    # align return a all zero matrix if failed to detect faces
                    # speed improved 7x by simply remove the py_func (from 14 sec/batch to 2 sec/batch on Tesla K40c)
                    # Test on TensorFlow dataset API: ~20.2 entries/sec on h1
                    # image_transformed = tf.py_func(lambda img: aligner.align(img), [image_decoded], tf.float32)
                    # image_transformed.set_shape([112, 96, 3])

                    # read preprocessed image directly from disk
                    image_transformed = image_decoded
                    image_transformed.set_shape([112, 96, 3])

                with tf.name_scope("image_normalization"):
                    # the implementation of normalization in 1704.08063 Sec 4.1
                    image_normalized = tf.div(tf.subtract(tf.cast(image_transformed, tf.float32), 127.5), 128)
                    # facenet use so-called "prewhiten"
                    # image_normalized = tf.image.per_image_standardization(image_resized)
                with tf.name_scope("data_augmentation"):
                    image_augmented = tf.image.random_flip_left_right(image_normalized)

                # image_augmented = tf.cond(tf.reduce_all(tf.equal(image_transformed, tf.zeros_like(image_transformed))),
                #                           lambda: image_transformed,
                #                           lambda: image_augmented)
                return image_augmented, tf.one_hot(label, depth=num_class)

            dataset = tf.data.Dataset.from_tensor_slices(
                (list(map(lambda item: item[0], data_list)),
                 list(map(lambda item: item[1], data_list))))

            dataset = dataset.prefetch(batch_size * 100)
            dataset = dataset.map(decode_data)
            # remove photos which failed to detect faces
            dataset = dataset.filter(lambda image, label: tf.reduce_all(tf.not_equal(image, tf.zeros_like(image))))

            if is_training:
                dataset = dataset.shuffle(10 * batch_size).repeat(epoch_num).batch(batch_size)
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
        images_data = list(
            map(lambda img_file_name: (os.path.join(cls_path, img_file_name), cls_name), images))
    return images_data


def get_fail_detect_files(path: str) -> set:
    """
    get file list which failed to detect faces
    :param path: the file path of fail file
    :return: image paths of failed images
    """
    if os.path.isdir(path):
        raise ValueError("Parameter path should be a file instead of a dic")
    with open(path, 'r') as file:
        lines = file.readlines()
        return set(map(lambda img_path: img_path.replace('\n', ''), lines))
    pass


def get_bounding_boxes(path, margin=44, img_size=None):
    """file format
            [offset_height, offset_width, target_height, target_width, confidence,
            left_eye_x, right_eye_x, nose_x, left_lip_x, right_lip_x,
            left_eye_y, right_eye_y, nose_y, left_lip_y, right_lip_y]"""

    if img_size is None:
        # image size by
        # https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m#L21
        img_size = [112, 96]
    if os.path.isdir(path):
        raise ValueError("Parameter path should be a file instead of a dic")

    bounding_boxes = {}
    with open(path, 'r') as file:
        bb_reader = csv.reader(file, delimiter=',')
        for row in bb_reader:
            # bounding box
            det = [float(item) for item in row[1:5]]
            x1 = np.maximum(det[0] - margin / 2, 0)
            y1 = np.maximum(det[1] - margin / 2, 0)
            x2 = np.minimum(det[2] + margin / 2, img_size[1])
            y2 = np.minimum(det[3] + margin / 2, img_size[0])

            # normalize bounding box to fit the requirement of tensorflow tf.image.crop_to_bounding_box
            # bb = (offset_height,
            #     offset_width,
            #     target_height,
            #     target_width)
            bb = (int(y1), int(x1), int(y2 - y1), int(x2 - x1))

            # key points
            key_points = [float(item) for item in row[6:16]]

            bounding_boxes[row[0]] = [bb, key_points]
        return bounding_boxes


def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)
    #     src_pts = np.array(src_pts).reshape((5,2))
    src_pts = np.array(src_pts).reshape((2, -1)).T

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = tools.cp2tform_np.get_similarity_transform_for_cv2(s.copy(), r.copy())

    # print(src_img.shape)
    # print(src_img)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    #     print(face_img.shape)
    return np.expand_dims(face_img, axis=2).astype(np.float32)
