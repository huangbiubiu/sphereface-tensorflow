# -*- coding: utf-8 -*-
from unittest import TestCase
import tools.cp2tform_np
import tools.cp2tform
import tensorflow as tf
import numpy as np

from align.mtcnn import detect_face
import skimage.io


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

    def test_get_similarity_transform_for_cv2(self):
        dst_pts = np.array([[30.2946, 51.6963],
                            [65.5318, 51.5014],
                            [48.0252, 71.7366],
                            [33.5493, 92.3655],
                            [62.7299, 92.2041]], dtype=np.float32)

        mtcnn_param = load_model()
        image_path = './images/002-l.jpg'
        image = skimage.io.imread(image_path)
        _, key_points = align(image, mtcnn_param)

        # reshape points
        dst_pts = np.reshape(dst_pts, (-1, 2), order='F')
        key_points = np.reshape(key_points, (-1, 2), order='F')

        # Set GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            np_result = tools.cp2tform_np.get_similarity_transform_for_cv2(src_pts=key_points.copy(), dst_pts=dst_pts.copy())

            tf_result = sess.run(tools.cp2tform.get_similarity_transform_for_cv2(src_pts=key_points.copy(), dst_pts=dst_pts.copy()))

            self.check_diff(np_result, tf_result)
            pass

    @staticmethod
    def check_diff(a: np.ndarray, b: np.ndarray, diff=1e-4):
        assert a.shape == b.shape, "Shape does not match"
        assert np.linalg.norm(a - b) / a.size < diff

    pass


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
    bounding_boxes, key_points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

    return bounding_boxes.tolist(), key_points
