# -*- coding: utf-8 -*-


from abc import ABCMeta, abstractmethod


class NerualNetwork(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def inference(self, images, num_class):
        pass
