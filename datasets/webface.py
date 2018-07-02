# -*- coding: utf-8 -*-
import csv
import cv2
import os
import tensorflow as tf
import tensorflow.contrib as contrib
import tools.cp2tform_np
import numpy as np

from tools.cp2tform import get_similarity_transform_for_cv2

# from https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m#L22
coord5point = np.array([[30.2946, 51.6963],
                        [65.5318, 51.5014],
                        [48.0252, 71.7366],
                        [33.5493, 92.3655],
                        [62.7299, 92.2041]], dtype=np.float32)


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


def load_data(data_dir, is_training, epoch_num, batch_size, param):
    data_dir = os.path.expanduser(data_dir)

    if 'image_size' in param:
        image_size = param['image_size']
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

    fail_detect_images = get_fail_detect_files(param['fail_path'])
    bounding_boxes = get_bounding_boxes(param['bounding_boxes'], margin=param['margin'])

    # remove images failed to detect faces
    data_list = filter(lambda item: item[0] not in fail_detect_images, data_list)
    # add bounding boxes to dataset
    data_list = list(map(lambda item: [item[0], item[1], bounding_boxes[item[0]]], data_list))

    # construct tensorflow dataset graph
    with tf.name_scope("load_webface"):
        def decode_data(image_path: str, label, bounding_box, key_points):
            """
            decode path to tensor
            :param image_path: path of image to be decoded
            :param label: training label of image
            :param bounding_box: detection infomation of image

            :return:
            """
            # read image from file
            image_file = tf.read_file(image_path)
            image_decoded = tf.image.decode_image(image_file)

            image_decoded.set_shape([100, 100, 1])

            # crop image
            with tf.name_scope("image_alignment"):
                '''order of key points:
                 [left_eye_x, right_eye_x, nose_x, left_lip_x, right_lip_x,
                 left_eye_y, right_eye_y, nose_y, left_lip_y, right_lip_y]'''
                # image_transformed = tf.py_func(alignment, [image_decoded, key_points], tf.float32)
                # image_transformed.set_shape([112, 96, 1])

                # key_pts = tf.transpose(tf.reshape(key_points, (2, -1)))
                key_pts = tf.py_func(lambda pts: np.reshape(pts, (-1, 2), order='F'), [key_points], tf.float32)
                key_pts.set_shape([5, 2])
                with tf.name_scope("convert_cv_transform_to_tf"):
                    transform_cv2 = get_similarity_transform_for_cv2(key_pts, coord5point)
                    transform_tf = tf.concat(axis=1,
                                             values=(tf.reshape(transform_cv2, (1, -1)),
                                                     tf.zeros((1, 2), dtype=tf.float32)))

                # image_transformed = contrib.image.transform(image_decoded, transform)
                image_transformed = tf.py_func(
                    lambda img, trans: np.expand_dims(cv2.warpAffine(img, trans, (image_size[1], image_size[0])),
                                                      axis=2),
                    [image_decoded, transform_cv2], tf.uint8)
                image_transformed = tf.cast(image_transformed, tf.float32)
                image_transformed.set_shape([112, 96, 1])
                # image_cropped = tf.image.crop_to_bounding_box(image_decoded,
                #                                               bounding_box[0],
                #                                               bounding_box[1],
                #                                               bounding_box[2],
                #                                               bounding_box[3])
                # image_resized = tf.image.resize_images(image_transformed, image_size)

            with tf.name_scope("image_normalization"):
                # the implementation of normalization in 1704.08063 Sec 4.1
                image_normalized = tf.div(tf.subtract(tf.cast(image_transformed, tf.float32), 127.5), 128)
                # facenet use so-called "prewhiten"
                # image_normalized = tf.image.per_image_standardization(image_resized)
            with tf.name_scope("data_augmentation"):
                # image_augmented = tf.image.random_flip_left_right(image_normalized)
                # CASIA-WebFace is already flipped
                image_augmented = image_normalized

            return image_augmented, tf.one_hot(label, depth=num_class)

        dataset = tf.data.Dataset.from_tensor_slices(
            (list(map(lambda item: item[0], data_list)),
             list(map(lambda item: item[1], data_list)),
             list(map(lambda item: item[2][0], data_list)),
             list(map(lambda item: item[2][1], data_list))))

        dataset = dataset.prefetch(batch_size)
        dataset = dataset.map(decode_data)

        if is_training:

            dataset = dataset.shuffle(10 * batch_size).repeat(epoch_num).batch(batch_size)
        else:
            dataset = dataset.shuffle(10 * batch_size).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next(), num_class
