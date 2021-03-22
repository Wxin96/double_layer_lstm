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
from mpl_toolkits.mplot3d import Axes3D

from macro.Mode import RangingMode


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
    traj = np.zeros((step_num, 3))
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
    for i in range(1, step_num):
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
    assert traj.shape == (len(traj), 3)
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
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
    nlos_record = np.zeros(shape=(len(traj)))
    # coordinate translation
    traj -= origin_coordinate
    # 重新生成NLOS情况，分阶段性
    # los情况 步数
    # nlos情况，步数
    env_num = 1  # >= 处于当前环境未结束
    env_mode = -1  # -1 => los, 0,1,...,n-1 基站序号
    # generator 3d ranging data
    for traj_idx in range(len(traj)):
        if env_num <= 0 and (mode == 1 or mode == RangingMode.NLOS):
            # 重新生成步数
            env_num = random.randint(5, 20)
            # 模式重新生成
            if random.random() < nlos_prob:
                env_mode = random.randint(0, len(anchors_location) - 1)
            else:
                env_mode = -1
            pass
        for anchor_idx in range(len(anchors_location)):
            if mode == 1 or mode == RangingMode.NLOS:
                if env_mode == anchor_idx:
                    ranging_data[traj_idx][anchor_idx] \
                        = generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], nlos_bias, nlos_sd)
                    # 进行了一步, nlos
                    env_num -= 1
                    nlos_record[traj_idx] = 1
                    print("nlos存在")
                else:
                    ranging_data[traj_idx][anchor_idx] \
                        = generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], 0, los_sd)
            else:
                ranging_data[traj_idx][anchor_idx] = \
                    generator_single_ranging(anchors_location[anchor_idx], traj[traj_idx], 0, los_sd)
        # los走了一步
        if env_mode == -1:
            env_num -= 1
    return ranging_data, traj, nlos_record


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


def generate_point_location_ranging(anchors_loc: np.ndarray, tag_loc: np.ndarray, nlos_prob, nlos_bias: float,
                                    nlos_sd: float, los_sd: float) -> np.ndarray:
    """
    返回单个点的测距数据。
    Args:
        anchors_loc: 基站坐标, 向量维度：（num_anchor，3）
        tag_loc: 标签坐标，（3，1）
        nlos_prob: nlos概率
        nlos_bias: 偏置
        los_sd: 标准差

    Returns:
        测距数据，（num_anchor，）
    """
    assert anchors_loc.shape == (len(anchors_loc), 3)
    assert tag_loc.shape == (3,)
    assert 0 <= nlos_prob <= 1
    point_ranging = np.zeros(len(anchors_loc))
    nlos_num = 0
    for i in range(len(point_ranging)):
        if random.random() < nlos_prob:
            point_ranging[i] = generator_single_ranging(anchors_loc[i], tag_loc, nlos_bias, nlos_sd)
            nlos_num += 1
        else:
            point_ranging[i] = generator_single_ranging(anchors_loc[i], tag_loc, 0, los_sd)
    # print("nlos次数为：" + str(nlos_num))
    return point_ranging


# 新增轨迹生成方式
def generator_3d_trajectory_2(step_num: int, length: float = 15.0, width: float = 6.0, high: float = 3.0,
                              random_loc_flag: bool = True, x_initial: float = 0.0,
                              y_initial: float = 0.0, z_initial: float = 0.0,
                              z_low: float = 0.0, z_high: float = 2.0,
                              speed: float = 1.5, delta_t: float = 0.2) -> np.ndarray:
    """
    考虑速度生成轨迹，目前是恒定速度直线方式前进。
    :param step_num: 总步长
    :param length: 三维空间长，unit: m
    :param width: 三维空间宽，unit：m
    :param high: 三维空间，unit：m
    :param random_loc_flag: True，随机位置；
    :param x_initial: random_loc=False, 初始位置x坐标，unit：m
    :param y_initial: 。。。
    :param z_initial: 。。。
    :param z_low: 随机位置，z的最小值
    :param z_high: 随机位置，z的最大值
    :param speed 行走速度，unit：m/s
    :param delta_t 采样间隔，unit: s
    :return: 返回轨迹
    """
    # param check
    if step_num <= 0:
        raise ValueError("输入参数：步长或步数有误！")
    if length <= 0 or width <= 0 or high <= 0:
        raise ValueError("输入参数：三维空间大小参数有误！")
    if x_initial < 0 or x_initial > length or y_initial < 0 or y_initial > width or z_initial < 0 or z_initial > high:
        raise ValueError("输入参数：初始位置有误！")

    # 初始位置
    def random_loc():
        x_random = length * random.random()
        y_random = width * random.random()
        z_random = z_high * random.random()
        return np.array([[x_random, y_random, z_random]])

    if random_loc_flag:
        loc = random_loc()
        # print(loc)
        x_initial = loc[0, 0]
        y_initial = loc[0, 1]
        z_initial = loc[0, 2]
    # 轨迹
    step_idx = 0
    traj = np.zeros(shape=(step_num, 3))
    traj[step_idx, :] = np.array([x_initial, y_initial, z_initial])
    # 轨迹生成
    while step_idx < step_num - 1:
        dest_loc = random_loc()
        if np.linalg.norm(dest_loc - traj[step_idx].reshape(1, 3)) < 5:
            continue
        tmp_traj = walk_line_a2b(traj[step_idx].reshape(1, 3), dest_loc, speed, delta_t)
        len_tmp_traj = len(tmp_traj)
        if step_idx + len_tmp_traj < step_num:
            traj[step_idx + 1:step_idx + len_tmp_traj + 1] = tmp_traj
            step_idx = step_idx + len_tmp_traj
        else:
            traj[step_idx + 1:step_num] = tmp_traj[0:step_num - step_idx - 1]
            step_idx = step_num - 1
        pass
    return traj
    pass


# 均衡速度直线行走
def walk_line_a2b(a_loc: np.ndarray, b_loc: np.ndarray, speed: float, delta_t: float) -> np.ndarray:
    """
    速度为 speed m/s, 从 a点 沿着直线 以恒定速度走到 b点
    :param a_loc: 起点，维数：（1,3）
    :param b_loc: 终点，维度，
    :param speed: 人行走速度, unit: m/s
    :param delta_t: 采样间隔, unit: s
    :return: 返回轨迹坐标，维度-（ ， 3）[含尾不含头]
    """
    assert a_loc.shape == (1, 3)
    assert b_loc.shape == (1, 3)
    # 计算各个轴速度
    direct = b_loc - a_loc
    dist = np.linalg.norm(direct)
    direct = direct / dist
    # print(direct)
    x_speed = speed * direct[0, 0]
    y_speed = speed * direct[0, 1]
    z_speed = speed * direct[0, 2]
    move_speed = np.array([x_speed, y_speed, z_speed])
    # 计算需要多少步
    step = math.ceil(dist / speed / delta_t)
    traj = np.zeros(shape=(step, 3))
    step_len = move_speed * delta_t
    for i in range(step):
        if i == 0:
            traj[i] = step_len + a_loc[0]
        else:
            traj[i] = traj[i - 1] + step_len
    # print(traj)
    return traj
    pass
