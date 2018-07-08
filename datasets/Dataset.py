# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Dataset():
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_data(self, dataset_path: str,
                  is_training: bool,
                  epoch_num: int,
                  batch_size: int,
                  data_param: dict) -> (tf.Tensor, int):
        """
        load data with TensorFlow dataset API
        :rtype: (tf.Tensor, int)
        :param dataset_path: the file path of dataset
        :param is_training: computation status: is training or not
        :param epoch_num: the number of epoch
        :param batch_size: the number of samples in each batch
        :param data_param: other parameters
        """
        pass
