# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import more_itertools as mit
import numpy as np
import skimage.io
import tensorflow as tf

from align.mtcnn.Aligner import Aligner


class AlignClass:
    def __init__(self, input_path, output_path, dataset_type):
        self.input_path = input_path
        self.output_path = output_path

        dataset_type = str.upper(dataset_type)
        if dataset_type == 'TRAIN':
            self.fail_to_detect = 'None'
        elif dataset_type == 'EVAL':
            self.fail_to_detect = 'RESIZE'
        else:
            raise ValueError(f'Not supported dataset mode: {dataset_type}')

        self.aligner = None

    def __call__(self, class_names):
        self.aligner = Aligner()

        if isinstance(class_names, str):
            class_names = [class_names]

        for class_name in class_names:
            if not os.path.exists(os.path.join(self.output_path, class_name)):
                os.makedirs(os.path.join(self.output_path, class_name))

            input_list = os.listdir(os.path.join(self.input_path, class_name))
            for image_file in input_list:
                image_full_path = os.path.join(self.input_path, class_name, image_file)
                image = skimage.io.imread(image_full_path)
                image_transformed = self.aligner.align(image, dtype=np.uint8, fail_to_detect=self.fail_to_detect)
                if image_transformed is None:
                    continue
                output_path = os.path.join(self.output_path, class_name, image_file)
                skimage.io.imsave(output_path, image_transformed)


def main(args, processing_num):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    clear_dir(output_dir)

    data_dir = os.path.expanduser(args.input_dir)
    face_classes = filter(lambda d: os.path.isdir(os.path.join(data_dir, d)), os.listdir(data_dir))

    pool = Pool(processing_num)
    pool.map(AlignClass(data_dir, output_dir, args.dataset_mode),
             chunks(face_classes, processing_num))

    pool.close()
    pool.join()


def clear_dir(folder: str):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def chunks(l, n):
    """Yield successive n part chunks from l."""
    return [list(c) for c in mit.divide(n, l)]


def progress(count, total, prefix='', suffix=''):
    sys.stdout.write('\r>> %s %.3f%% %s' % (prefix, float(count) / float(total) * 100.0, suffix))
    sys.stdout.flush()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')

    parser.add_argument('dataset_mode', type=str, help='Directory with aligned face thumbnails.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.WARN)
    main(parse_arguments(sys.argv[1:]), 20)
