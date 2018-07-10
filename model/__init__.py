# -*- coding: utf-8 -*-
from enum import Enum, unique


@unique
class GraphType(Enum):
    TRAIN = 0
    TEST = 1
    EVAL = 2
