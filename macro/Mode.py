#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : Mode.py
@ide    : PyCharm
@time   : 21/3/20 8:56
@desc   ：
"""
from enum import Enum


class RangingMode(Enum):
    """
    测距模式
    """
    LOS = 0
    NLOS = 1
    pass
