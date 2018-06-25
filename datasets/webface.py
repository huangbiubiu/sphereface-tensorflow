# -*- coding: utf-8 -*-
import csv
import os
import tensorflow as tf
import numpy as np


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
    if img_size is None:
        img_size = [100, 100]
    if os.path.isdir(path):
        raise ValueError("Parameter path should be a file instead of a dic")

    bounding_boxes = {}
    with open(path, 'r') as file:
        bb_reader = csv.reader(file, delimiter=',')
        for row in bb_reader:
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

            bounding_boxes[row[0]] = bb
        return bounding_boxes


def load_data(data_dir, is_training, epoch_num, batch_size, param):
    data_dir = os.path.expanduser(data_dir)

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
        def decode_data(image_path, label, bounding_box):
            # read image from file
            image_file = tf.read_file(image_path)
            image_decoded = tf.image.decode_image(image_file)

            image_decoded.set_shape([100, 100, 1])

            # crop image
            with tf.name_scope("image_cropping"):
                image_cropped = tf.image.crop_to_bounding_box(image_decoded,
                                                              bounding_box[0],
                                                              bounding_box[1],
                                                              bounding_box[2],
                                                              bounding_box[3])
                image_resized = tf.image.resize_images(image_cropped, [100, 100])

            with tf.name_scope("image_normalization"):
                # the implementation of normalization in 1704.08063 Sec 4.1
                image_normalized = tf.div(tf.subtract(tf.cast(image_resized, tf.float32), 127.5), 128)
                # facenet use so-called "prewhiten"
                # image_normalized = tf.image.per_image_standardization(image_resized)
            with tf.name_scope("data_augmentation"):
                image_augmented = tf.image.random_flip_left_right(image_normalized)

            return image_augmented, tf.one_hot(label, depth=num_class)

        dataset = tf.data.Dataset.from_tensor_slices(
            (list(map(lambda item: item[0], data_list)),
             list(map(lambda item: item[1], data_list)),
             list(map(lambda item: item[2], data_list))))

        dataset = dataset.prefetch(batch_size)
        dataset = dataset.map(decode_data)

        if is_training:

            dataset = dataset.shuffle(10 * batch_size).repeat(epoch_num).batch(batch_size)
        else:
            dataset = dataset.shuffle(10 * batch_size).batch(batch_size)

        return dataset.make_one_shot_iterator().get_next(), num_class


def check_dim(path):
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()

    import skimage.io
    import numpy as np

    data_dir = os.path.expanduser(path)

    # construct dataset with python code
    face_classes = os.listdir(data_dir)
    class2index = {k: v for v, k in enumerate(face_classes)}
    data_list = list(map(lambda cls: list_images(data_dir, cls, class2index), face_classes))
    # flat the list
    data_list = [item for sublist in data_list for item in sublist]

    iter_count = 0
    printProgressBar(iter_count, len(data_list))
    depth_count = [0, 0]
    for image_path, label in data_list:
        img_file = skimage.io.imread(image_path)
        if len(img_file.shape) == 2:
            depth_count[0] += 1
        elif img_file.shape[2] == 3:
            depth_count[1] += 1
        else:
            raise ValueError(f"{image_path} with shape {depth}")
        iter_count = iter_count + 1
        printProgressBar(iter_count, len(data_list),
                         suffix=f"(100,100): {depth_count[0]}, (100,100,3): {depth_count[1]}")


if __name__ == '__main__':
    fail_path = '/home/hyh/datasets/CASIA-WebFace/Normalized_Faces/webface/fail.txt'
    bb_path = '/home/hyh/datasets/CASIA-WebFace/Normalized_Faces/webface/bounding_boxes.txt'

    data_dir = '/home/hyh/datasets/CASIA-WebFace/Normalized_Faces/webface/100'

    data, _ = load_data(data_dir, True, None, 256, {'fail_path': fail_path, 'bounding_boxes': bb_path})

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(data)
