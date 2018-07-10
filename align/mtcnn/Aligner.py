# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf

from align.mtcnn import detect_face
from tools.cp2tform_np import get_similarity_transform_for_cv2

TENSORFLOW = 0
NUMPY = 1


class Aligner:
    def __init__(self, image_size, impl_ver=TENSORFLOW):
        self.mtcnn_param = self.load_model()
        self.coord5point = np.array([[30.2946, 51.6963],
                                     [65.5318, 51.5014],
                                     [48.0252, 71.7366],
                                     [33.5493, 92.3655],
                                     [62.7299, 92.2041]], dtype=np.float32)
        self.image_size = image_size
        self.impl_ver = impl_ver

    @staticmethod
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

    def detect_faces(self, image, mtcnn_param):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        pnet = mtcnn_param['pnet']
        rnet = mtcnn_param['rnet']
        onet = mtcnn_param['onet']

        if len(np.shape(image)) == 2:  # expand gray scale image to RGB
            image = np.stack((image, image, image), axis=2)
        bounding_boxes, key_points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) > 1:
            # select the bounding box with largest area
            def area(bb):
                x1, y1, x2, y2 = bb[:4]
                return (x2 - x1) * (y2 - y1)

            bounding_box, key_point = max(zip(bounding_boxes, key_points.T.tolist()), key=lambda item: area(item[0]))
            return np.reshape(bounding_box, (1, 5)).tolist(), np.reshape(key_point, (10, 1)).tolist()
        elif len(bounding_boxes) == 0:
            return [], []

        return bounding_boxes.tolist(), key_points.tolist()

    def detect(self, image):
        return self.detect_faces(image, self.mtcnn_param)

    def get_coord_points(self):
        return self.coord5point.copy()

    def align(self, image, dtype=np.float32):
        """
        align image
        If failed to detect faces in input image, a image with all zero value will be returned.
        :param image: original photo
        :param dtype: data type to return
        :return: transformed image
        """
        bounding_boxes, key_points = self.detect(image)
        if len(bounding_boxes) == 0:
            return np.zeros((*self.image_size, 3)).astype(dtype)
        key_points = np.reshape(key_points, (-1, 2), order='F')

        if len(np.shape(image)) == 2:  # expand gray scale photo
            image = np.stack([image] * 3, axis=2)

        transform_cv2 = get_similarity_transform_for_cv2(key_points, self.get_coord_points())
        image_transformed = cv2.warpAffine(image, transform_cv2, (self.image_size[1], self.image_size[0]))
        if len(np.shape(image_transformed)) == 2:  # expand gray scale photo to RGB
            image_transformed = np.expand_dims(image_transformed, axis=2)

        return image_transformed.astype(dtype)


def test():
    import skimage.io
    # test Align class

    aligner = Aligner([112, 96])

    imgs = []
    for _ in range(10):
        image_path = '/home/hyh/datasets/CASIA-WebFace/Small_Piece_For_Easy_Download/CASIA-WebFace/0005045/188.jpg'
        image = skimage.io.imread(image_path)
        transformed_image = aligner.align(image=image, dtype=np.uint8)
        imgs.append(transformed_image)

    pass
