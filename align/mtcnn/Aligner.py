# -*- coding: utf-8 -*-
import cv2

from align.mtcnn.align_dataset import load_model, align
import numpy as np

from tools.cp2tform_np import get_similarity_transform_for_cv2


class Aligner:
    def __init__(self, image_size):
        self.mtcnn_param = load_model()
        self.coord5point = np.array([[30.2946, 51.6963],
                                     [65.5318, 51.5014],
                                     [48.0252, 71.7366],
                                     [33.5493, 92.3655],
                                     [62.7299, 92.2041]], dtype=np.float32)
        self.image_size = image_size

    def detect(self, image):
        return align(image, self.mtcnn_param)

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
