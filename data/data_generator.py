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
import math
import matplotlib.pyplot as plt


def generator_3d_trajectory(step_num: int, step_len: float = 0.55, step_mode: int = 0,
                            length: float = 15.0, width: float = 6.0, high: float = 3.0,
                            random_loc: bool = True, x_initial: float = 0.0,
                            y_initial: float = 0.0, z_initial: float = 0.0,
                            x_direct_prob: float = 0.3, y_direct_prob: float = 0.3,
                            z_direct_prob: float = 0.2, static_prob: float = 0.2,
                            z_low: float = 0.0, z_high: float = -1.0):
    """
    生成 3d 行走轨迹, unit: m(米).
    Args:
        step_num: 步数
        step_len: 步长，内部取正太分布，step_len为均值，0.1为标准差
        step_mode: 行走模式，0-x，y，z直角坐标系方向，1-球坐标系方向
        length: 三维场景的长
        width: 三维场景的宽
        high: 三维场景的高
        random_loc: True-随机位置，False-自定义初始位置
        x_initial: x坐标初始位置
        y_initial: y坐标初始位置
        z_initial: z坐标初始位置
        x_direct_prob: x坐标方向行走的概率
        y_direct_prob: y坐标方向行走的概率
        z_direct_prob: z坐标方向行走的概率
        static_prob: 静止不走的概率
        z_low: z轴的最低，考虑人体行走
        z_high: z轴的最高，考虑人体行走
    Returns:
        返回轨迹坐标，维度（step_len + 1, 3）维度
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
    if z_low < 0 or z_high > high:
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
    def walk_mode(mode: int):
        """
        两种不同的模式进行随机漫步。
        Args:
            mode:
                0: x、y、z轴方向进行随机漫步（直角坐标系）
                1: 任意方向进行随机漫步（球坐标系）
                    x = r * sin(theta) * cos(phi)
                    y = r * sin(theta) * sin(phi)
                    z = r * cos(theta)
        Returns：
            x_incr, y_incr, z_incr: 三维增量
        """
        x_incr = y_incr = z_incr = 0
        walk_direction = random.random()
        if walk_direction > z_direct_prob:
            return x_incr, y_incr, z_incr
        if mode == 0:
            if walk_direction <= x_direct_prob:
                x_incr = random.gauss(step_len, 0.1) * random.choice([-1, 1])
            elif walk_direction <= y_direct_prob:
                y_incr = random.gauss(step_len, 0.1) * random.choice([-1, 1])
            elif walk_direction <= z_direct_prob:
                z_incr = random.gauss(step_len / 2, 0.1) * random.choice([-1, 1])
        elif mode == 1:
            r = random.gauss(step_len, 0.1)
            ratio = y_direct_prob / (z_direct_prob - y_direct_prob)  # (x, y) 方向 与 z 方向的比
            ratio_radian = math.atan(ratio)
            theta = random.uniform(ratio_radian, math.pi - ratio_radian)
            phi = random.uniform(0, 2 * math.pi)
            x_incr = r * math.sin(theta) * math.cos(phi)
            y_incr = r * math.sin(theta) * math.sin(phi)
            z_incr = r * math.cos(theta)

        return x_incr, y_incr, z_incr

    # iteration, random walk
    for i in range(1, step_num + 1):
        x_cur_loc, y_cur_loc, z_cur_loc = traj[i - 1]
        while True:
            x_tmp_loc, y_tmp_loc, z_tmp_loc = walk_mode(step_mode)
            x_tmp_loc += x_cur_loc
            y_tmp_loc += y_cur_loc
            z_tmp_loc += z_cur_loc
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
        return False
    if y < y_start or y > y_end:
        return False
    if z < z_start or z > z_end:
        return False
    return True


def draw_trajectory(traj: np.ndarray):
    """
    绘制三维空间轨迹。
    Args:
        traj: 轨迹坐标
    Returns:
        无
    """
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


def generator_3d_ranging_data(traj: np.ndarray, anchors_location: np.ndarray, origin_coordinate: np.ndarray,
                              los_sd: float, nlos_bias: float, nlos_sd: float,
                              mode: int = 0, nlos_prob: float = 0.2) -> (np.ndarray, np.ndarray):
    """
    根据轨迹生成基站测距数据。
    unit: m
    Args:
        traj: 轨迹信息，维度（time_step, 3）, time_step-步长
        anchors_location: 三维基站坐标，维度（num_anchor, 3）, num_anchor-基站个数
        origin_coordinate: 坐标原点在三维空间的位置，进行坐标平移
        los_sd: LOS环境下标准差
        nlos_bias: nlos环境下偏差
        nlos_sd: nlos环境下标准差
        mode:  0-los条件，1-nlos环境(根据概率，基站有可能出现nlos情况)
        nlos_prob: nlos出现的概率
    Returns:
        基站测距数据，维数（time_step, num_anchor）,坐标平移后的轨迹坐标
    """
    # check
    assert traj.shape == (len(traj), 3)
    assert anchors_location.shape == (len(anchors_location), 3)
    assert origin_coordinate.shape == (3,)
    assert los_sd >= 0 and nlos_sd >= 0 and 0 <= nlos_prob <= 1
    # init
    ranging_data = np.zeros(shape=(len(traj), len(anchors_location)))
    # coordinate translation
    traj -= origin_coordinate
    # generator 3d ranging data
    for traj_idx in range(len(traj)):
        for anchor_idx in range(len(anchors_location)):
            if mode == 1:
                ranging_data[traj_idx][anchor_idx] = \
                    generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], nlos_bias, nlos_sd) \
                        if random.random() < nlos_prob \
                        else generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], 0, los_sd)
            else:
                ranging_data[traj_idx][anchor_idx] = \
                    generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], 0, los_sd)
    return ranging_data, traj


def generator_single_ranging(anchor_loc: np.ndarray, tag_loc: np.ndarray, bias: float, sd: float) -> float:
    """
    根据基站和标签位置，添加偏差和正态分布随机误差，模拟生成测距数据.
    参数的单位，unit:m
    :param anchor_loc: 基站位置，向量维度-（3,）
    :param tag_loc: 标签位置，向量维度-（3,）
    :param bias: nlos偏置误差
    :param sd: 正态分布的标准差
    :return: 单次测距距离,unit: m
    """
    assert anchor_loc.shape == (3,)
    assert tag_loc.shape == (3,)
    dist = np.linalg.norm(anchor_loc - tag_loc)
    dist += random.gauss(bias, sd)
    return dist
