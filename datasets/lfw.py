# -*- coding: utf-8 -*-
import multiprocessing
import os
import random
import re

import numpy as np
import tensorflow as tf

from align.mtcnn.Aligner import Aligner
from datasets.Dataset import Dataset


class LFW(Dataset):
    def load_data(self, dataset_path, is_training, epoch_num, batch_size, data_param):
        if 'image_size' in data_param:
            image_size = data_param['image_size']
            if isinstance(image_size, int):
                image_size = [image_size, image_size]
        else:
            image_size = [112, 96]
        self.aligner = Aligner(image_size)

        with tf.name_scope("load_lfw"):
            def decode_data(image_path, label):
                # read image from file
                image_file = tf.read_file(image_path)
                image_decoded = tf.image.decode_image(image_file)
                image_decoded.set_shape([112, 96, 3])

                with tf.name_scope("image_alignment"):
                    # image_transformed = tf.py_func(lambda img: self.aligner.align(img), [image_decoded], tf.float32)
                    # image_transformed.set_shape([112, 96, 3])

                    image_transformed = image_decoded
                with tf.name_scope("image_normalization"):
                    # image_transformed = tf.cond(
                    #     tf.reduce_all(tf.equal(image_transformed, tf.zeros_like(image_transformed))),
                    #     lambda: tf.image.resize_images(image_decoded, image_size),
                    #     lambda: image_transformed)

                    # the implementation of normalization in 1704.08063 Sec 4.1
                    image_normalized = tf.div(tf.subtract(tf.cast(image_transformed, tf.float32), 127.5), 128)
                    # facenet use so-called "prewhiten"
                    # image_normalized = tf.image.per_image_standardization(image_resized)
                with tf.name_scope("data_augmentation"):
                    # image_augmented = tf.image.random_flip_left_right(image_normalized)
                    # CASIA-WebFace is already flipped
                    image_augmented = image_normalized

                return image_augmented, label

            if is_training:
                raise NotImplementedError("Training on LFW dataset is not supported.")
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (list(map(lambda item: item[0], self.flatten_pairs)),
                     list(map(lambda item: item[0], self.flatten_pairs))))
                dataset = dataset.prefetch(batch_size * 5)
                dataset = dataset.map(decode_data)

                dataset = dataset.batch(batch_size)

                return dataset.make_one_shot_iterator().get_next(), -1

        pass

    def __init__(self, data_path, pair_path=None):
        """
        load LFW dataset
        :param data_path: lfw dataset directory
        :param pair_path: the path of pairs.txt
        """
        self.path = os.path.expanduser(data_path)
        if pair_path is None:
            pair_path = os.path.expanduser(os.path.join(data_path, "pairs.txt"))

        if not os.path.exists(pair_path):
            raise ValueError(f"pairs.txt can not be found in {pair_path}")
        if not os.path.exists(data_path):
            raise ValueError(f"directory {data_path} does not exist")

        # read pairs.txt
        with open(pair_path, 'r') as pairs_file:
            """
            Example of pairs.txt:
            Abel_Pacheco    1       4
            Slobodan_Milosevic      2       Sok_An  1
            """
            lines = pairs_file.readlines()[1:]  # ignore the first line
            lines = list(map(lambda line: re.split(r'\s', str.replace(line, '\n', '')), lines))
            self.pairs = lines

            self.flatten_pairs = map(lambda p: self.pair2path(p, with_label=False), self.pairs)
            self.flatten_pairs = [item for sublist in self.flatten_pairs for item in sublist]

        # load dataset structure
        persons = filter(lambda file: os.path.isdir(file),
                         map(lambda item: os.path.join(data_path, item), os.listdir(data_path)))
        dataset = {}
        for p in persons:
            photos = list(map(lambda photo: os.path.join(p, photo), os.listdir(p)))
            dataset[p] = photos

        self.dataset = dataset

        # instance a Aligner to alignment photos
        self.aligner = None
        pass

    @staticmethod
    def get_image_path(basepath, name, index, ext='jpg'):
        filename = f"{name}_{str(index).zfill(4)}.{ext}"
        return os.path.join(basepath, name, filename), name

    def pair2path(self, pair, with_label=False):
        pair_path = []
        if len(pair) == 4:
            pair_path.append(self.get_image_path(self.path, pair[0], pair[1]))
            pair_path.append(self.get_image_path(self.path, pair[2], pair[3]))

            if with_label:
                pair_path.append(False)
        elif len(pair) == 3:
            pair_path.append(self.get_image_path(self.path, pair[0], pair[1]))
            pair_path.append(self.get_image_path(self.path, pair[0], pair[2]))
            if with_label:
                pair_path.append(True)
        else:
            raise ValueError(f"Illegal entry {pair}")

        if with_label:
            return pair_path[0], pair_path[1], pair_path[2]
        else:
            return pair_path[0], pair_path[1]

    def evaluation(self, embeddings: dict):
        """
        do 10-fold evaluation :param get_threshold: a callable object with (eval_pairs, simlarity) as parameters and
        return the best threshold :param embeddings: a embeddings list for lookup. storage file name as keys and
        corresponding embedding as values :param similarity: a callable object. receive a embedding array as input (
        with shape (N, 2)) and threshold returned by get threshold, calculate the similarity between two embeddings
        at each row, return a bool array with shape (N, )
        """
        pairs = self.pairs.copy()
        # expand pairs to path
        pairs = list(map(lambda p: self.pair2path(p, with_label=True), pairs))
        random.shuffle(pairs)

        chunk_size = len(pairs) // 10
        # acc_list = []
        # for i in range(10):
        #     val_pairs = pairs[i * chunk_size: (i + 1) * chunk_size]
        #     eval_pairs = pairs[:i * chunk_size] + pairs[(i + 1) * chunk_size:]
        #
        #     threshold = get_threshold(val_pairs, similarity, embeddings)
        #     acc = similarity(embeddings, eval_pairs, threshold)
        #     acc_list.append(acc)
        pool = multiprocessing.Pool(10)
        acc_list = pool.map(FoldEval(pairs, chunk_size, embeddings),
                            range(10))

        return np.mean(acc_list)


class FoldEval:
    def __init__(self, pairs, chunk_size, embeddings):
        self.pairs = pairs
        self.chunk_size = chunk_size

        self.embeddings = embeddings

    def __call__(self, i):
        val_pairs = self.pairs[i * self.chunk_size: (i + 1) * self.chunk_size]
        eval_pairs = self.pairs[:i * self.chunk_size] + self.pairs[(i + 1) * self.chunk_size:]

        threshold = self.get_threshold(val_pairs, self.embeddings)
        acc = self.cos_similarity(self.embeddings, eval_pairs, threshold)
        return acc

    def get_threshold(self, val_pairs, embeddings, thr_num=10000):
        # numpy implementation for
        # https://github.com/wy1iu/sphereface/blob/master/test/code/evaluation.m#L115
        thresholds = np.arange(-thr_num, thr_num, 1) / thr_num

        scores = np.array(list(map(lambda p: np.sum(embeddings[p[0][0]] * embeddings[p[1][0]]), val_pairs)))
        labels = list(map(lambda p: p[2], val_pairs))

        acc = list(map(lambda t: np.sum(labels == (scores > t)) / len(val_pairs), thresholds))
        return np.mean(thresholds[acc == np.max(acc)])

    def cos_similarity(self, emb: dict, eval_pairs, threshold):
        # numpy implementation for
        # https://github.com/wy1iu/sphereface/blob/master/test/code/evaluation.m#L69
        compare_result = list(map(lambda p: np.sum(emb[p[0][0]] * emb[p[1][0]]) > threshold == p[2], eval_pairs))

        return np.sum(compare_result) / len(compare_result)

        pass


if __name__ == '__main__':
    lfw = LFW('/home/hyh/datasets/lfw')
    lfw.evaluation(None, None, None)
