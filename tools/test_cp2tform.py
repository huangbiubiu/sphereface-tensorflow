# -*- coding: utf-8 -*-
from unittest import TestCase
import tools.cp2tform_np
import tools.cp2tform
import tensorflow as tf
import numpy as np


class TestCp2tform(TestCase):
    def test_tformfwd(self):
        # Set GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            K = np.random.randint(3, 20)
            trans = np.random.random((3, 3)).astype(np.float32)
            uv = np.random.random((K, 2)).astype(np.float32)
            np_result = tools.cp2tform_np.tformfwd(trans=trans, uv=uv)

            tf_result = sess.run(tools.cp2tform.tformfwd(trans=trans, uv=uv))

            assert np.abs(np.sum(np_result - tf_result)) < 10e-6
            pass

        pass

    def test_tforminv(self):
        # Set GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            K = np.random.randint(3, 20)
            trans = np.random.random((3, 3)).astype(np.float32)
            uv = np.random.random((K, 2)).astype(np.float32)

            np_result = tools.cp2tform_np.tforminv(trans=trans.copy(), uv=uv.copy())

            tf_result = sess.run(tools.cp2tform.tforminv(trans=trans.copy(), uv=uv.copy()))

            self.check_diff(np_result, tf_result)

    def test_findNonreflectiveSimilarity(self):
        # Set GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            K = np.random.randint(3, 20)
            uv = (100 * np.random.random((K, 2))).astype(np.float64)
            xy = (100 * np.random.random((K, 2))).astype(np.float64)

            np_result = tools.cp2tform_np.findNonreflectiveSimilarity(xy=xy.copy(), uv=uv.copy())

            tf_result = sess.run(tools.cp2tform.findNonreflectiveSimilarity(xy=xy.copy(), uv=uv.copy()))

            self.check_diff(np_result[0], tf_result[0])
            self.check_diff(np_result[1], tf_result[1])
            pass
        pass

    def test_findSimilarity(self):
        # Set GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            K = np.random.randint(3, 20)
            uv = (100 * np.random.random((K, 2))).astype(np.float32)
            xy = (100 * np.random.random((K, 2))).astype(np.float32)

            np_result = tools.cp2tform_np.findSimilarity(xy=xy.copy(), uv=uv.copy())

            tf_result = sess.run(tools.cp2tform.findSimilarity(xy=xy.copy(), uv=uv.copy()))

            self.check_diff(np_result[0], tf_result[0])
            self.check_diff(np_result[1], tf_result[1])
            pass

    @staticmethod
    def check_diff(a: np.ndarray, b: np.ndarray, diff=1e-4):
        assert a.shape == b.shape, "Shape does not match"
        assert np.linalg.norm(a - b) / a.size < diff

    pass
