# -*- coding: utf-8 -*-


from abc import ABCMeta, abstractmethod


class NerualNetwork(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def inference(self, images, num_class, param):
        pass

    @staticmethod
    def get_image_size(param: dict) -> list:
        """
        get the image size to construct the network
        :param param: parameter dictionary
        :return: A 1-D list of 2 elements: new_height, new_width. The fixed size for the input images.
        """
        if 'image_size' in param:
            if isinstance(param['image_size'], int):
                image_size = [param['image_size'], param['image_size']]
            elif isinstance(param['image_size'], list):
                if len(param['image_size']) != 2:
                    raise ValueError(f"Illegal parameter image_size: {param['image_size']}")
                image_size = param['image_size']
            else:
                raise ValueError(f"Illegal parameter image_size: {param['image_size']}")
        else:
            image_size = [32, 32]

        return image_size
