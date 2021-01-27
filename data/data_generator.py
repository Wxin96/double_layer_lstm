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
import matplotlib.pyplot as plt


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
    # 概率累计到 [0, 1] 区间上
    y_direct_prob += x_direct_prob
    z_direct_prob += y_direct_prob
    static_prob += z_direct_prob
    traj = np.zeros((step_num + 1, 3))
    traj[0] = x_initial, y_initial, z_initial

    # 行走模式
    def walk_mode(mode: int) -> np.ndarray:
        """
        两种不同的模式进行随机漫步。
        Args:
            mode:
                0: x、y、z轴方向进行随机漫步
                1: 任意方向进行随机漫步
        Returns：
            x_incr, y_incr, z_incr: 三维增量
        """
        x_incr = y_incr = z_incr = 0
        if mode == 0:
            walk_direction = random.random()
            if walk_direction <= x_direct_prob:
                x_incr = np.random.normal(step_len, 0.1) * random.choice([-1, 1])
            elif walk_direction <= y_direct_prob:
                y_incr = np.random.normal(step_len, 0.1) * random.choice([-1, 1])
            elif walk_direction <= z_direct_prob:
                z_incr = np.random.normal(step_len, 0.1) * random.choice([-1, 1])
        elif mode == 1:
            pass

        return x_incr, y_incr, z_incr

    # iteration, random walk
    for i in range(1, step_num + 1):
        x_cur_loc, y_cur_loc, z_cur_loc = traj[i - 1]
        while True:
            # TODO: 死循环
            # print("iteration")
            x_tmp_loc, y_tmp_loc, z_tmp_loc = walk_mode(0)
            x_tmp_loc += x_cur_loc
            y_tmp_loc += y_cur_loc
            z_tmp_loc += y_cur_loc
            print(x_tmp_loc)
            print(y_tmp_loc)
            print(z_tmp_loc)
            if position_check(0, length, 0, width, z_low, z_high, x_tmp_loc, y_tmp_loc, z_tmp_loc):
                traj[i] = x_tmp_loc, y_tmp_loc, z_tmp_loc
                break

    return traj


def position_check(x_start: float, x_end: float,
                   y_start: float, y_end: float,
                   z_start: float, z_end: float,
                   x: float, y: float, z: float) -> bool:
    """
    移动位置校验，避免出现出界。
    Args:
        x_start: x坐标初始位置
        x_end: x坐标终止位置
        y_start: y坐标初始位置
        y_end: y坐标终止位置
        z_start: z坐标初始位置
        z_end: z坐标终止位置
        x: 移动的位置 x 坐标
        y: 移动位置 y 坐标
        z: 移动位置 z 坐标
    Returns:
        合法返回 True，不合法返回 False。
    """
    if x < x_start or x > x_end:
        print(type(x))
        print(type(x_start))
        return False
    if y < y_start or y > y_end:
        print(type(y))
        print(type(y_start))
        return False
    if z < z_start or z > z_end:
        print(type(z))
        print(type(z_start))
        return False
    return True


def draw_trajectory(traj: np.ndarray):
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    # new a figure and set it into 3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # set figure information
    ax.set_title("3D_Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # draw the figure, the color is r = read
    figure = ax.plot(x, y, z, c='r')

    plt.show()


def generator_3d_anchor_location(length: int, width: int, high: int, num_anchor: int):
    anchor_location = np.array()
