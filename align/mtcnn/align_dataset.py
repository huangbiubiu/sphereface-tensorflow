# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import skimage.io
import tensorflow as tf

from align.mtcnn import detect_face
from datasets.webface import list_images


def load_model():
    # mtcnn session
    mtcnn_param = {}
    mtcnn_graph = tf.Graph()
    mtcnn_param['graph'] = mtcnn_graph
    with mtcnn_graph.as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        mtcnn_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                      log_device_placement=False),
                                graph=mtcnn_graph)
        mtcnn_param['sess'] = mtcnn_sess
        with mtcnn_sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(mtcnn_sess, None)
            mtcnn_graph.finalize()
            mtcnn_param['pnet'] = pnet
            mtcnn_param['rnet'] = rnet
            mtcnn_param['onet'] = onet

    return mtcnn_param


def align(image, mtcnn_param):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    pnet = mtcnn_param['pnet']
    rnet = mtcnn_param['rnet']
    onet = mtcnn_param['onet']

    image = np.stack((image, image, image), axis=2)
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

    return bounding_boxes.tolist()


def flatten(l: list) -> list:
    return [str(item) for sublist in l for item in sublist]


def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.expanduser(args.input_dir)
    face_classes = os.listdir(data_dir)
    data_list = list(map(lambda cls: list_images(data_dir, cls, None), face_classes))
    dataset = [item for sublist in data_list for item in sublist]

    mtcnn_param = load_model()

    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes.txt')
    fail_filename = os.path.join(output_dir, 'fail.txt')

    with open(bounding_boxes_filename, "w") as text_file, open(fail_filename, "w") as fail_file:
        count = 0
        fail_count = 0
        for image_path, _ in dataset:
            image = skimage.io.imread(image_path)
            bounding_boxes = align(image, mtcnn_param)

            count += 1
            if len(bounding_boxes) == 0:
                fail_count += 1
                fail_file.write(f"{image_path}\n")
            else:
                info = f"{image_path},{','.join(flatten(bounding_boxes))}\n"
                text_file.write(info)

            progress(count, len(dataset),
                     suffix=f'fail:count:total {fail_count}:{count}:{len(dataset)} Processing: {image_path}')


def progress(count, total, prefix='', suffix=''):
    sys.stdout.write('\r>> %s %.3f%% %s' % (prefix, float(count) / float(total) * 100.0, suffix))
    sys.stdout.flush()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
