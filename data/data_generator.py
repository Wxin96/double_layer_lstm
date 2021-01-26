#!/user/bin/env python
# coding=utf-8
"""
@project : double_layer_lstm
@author  : apollo
@file   : data_generator.py
@ide    : PyCharm
@time   : 2021-01-23 11:07:12
@desc    ：生成三维空间的轨迹
"""
import numpy as np
import random


def generator_3d_trajectory(step_num: int, step_len: float = 0.55,
                            length: float = 15.0, width: float = 6.0, high: float = 3.0,
                            random_loc: bool = False, x_initial: float = 0.0,
                            y_initial: float = 0.0, z_initial: float = 0.0,
                            x_direct_prob: float = 0.3, y_direct_prob: float = 0.3,
                            z_direct_prob: float = 0.2, static_prob: float = 0.2,
                            z_low: float = 0.0, z_high: float = -1.0):
    """
    unit: m(米)
    """
    # param check
    if step_num <= 0 or step_len <= 0:
        raise ValueError("输入参数：步长或步数有误！")
    if length <= 0 or width <= 0 or high <= 0:
        raise ValueError("输入参数：三维空间大小参数有误！")
    if x_initial < 0 or x_initial > length or y_initial < 0 or y_initial > width or z_initial < 0 or z_initial > high:
        raise ValueError("输入参数：初始位置有误！")
    if x_direct_prob < 0 or y_direct_prob < 0 or z_direct_prob < 0 or static_prob < 0 or x_direct_prob + y_direct_prob \
            + z_direct_prob + static_prob != 1:
        raise ValueError("输入参数：运动概率有误！")
    if z_low < 0:
        raise ValueError("输入参数：z的范围有误！")
    # param init
    if z_high < 0:
        z_high = high
    if random_loc:
        x_initial = length * random.random()
        y_initial = width * random.random()
        z_initial = z_high * random.random()
    y_direct_prob += x_direct_prob
    z_direct_prob += y_direct_prob
    static_prob += z_direct_prob
    traj = np.zeros((step_num + 1, 3))
    traj[0] = x_initial, y_initial, z_initial
    # iteration, random walk
    for i in range(1, step_num + 1):
        walk_direction = random.random()



def generator_3d_anchor_location(length: int, width: int, high: int, num_anchor: int):
    anchor_location = np.array();
