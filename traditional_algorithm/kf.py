#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : kf.py
@ide    : PyCharm
@time   : 2021-02-24 10:21:45
@desc   ：卡尔曼滤波，位置观测
"""
import numpy as np


class KF:
    """
    卡尔曼滤波，
        状态，位置、速度
        观测：标签位置
    """
    def __init__(self, spatial_dimension: int, delta_t: float, Q: np.ndarray, R: np.ndarray, P_init: np.ndarray):
        """
        扩展卡尔曼滤波初始化
        Args:
            spatial_dimension: 测距维数
            delta_t: 间隔时间
            Q: 过程噪声【方差】 维度-（2*spatial_dimension，2*spatial_dimension）
            R: 测量噪声【方差】 维度-（num_anchor，num_anchor）
            P_init: 预测估计协方差矩阵 初始值 维度-（2*spatial_dimension，2*spatial_dimension）
        """
        # 判断
        assert Q.shape == (2 * spatial_dimension, 2 * spatial_dimension)
        assert R.shape == (spatial_dimension, spatial_dimension)
        assert P_init.shape == (2 * spatial_dimension, 2 * spatial_dimension)

        # 基本参数
        self.__spatial_dimension = spatial_dimension

        # 扩展卡尔曼滤波参数
        self.__x = np.zeros(shape=(2 * spatial_dimension, 1))  # x初始化
        self.__P = P_init
        self.__y = np.zeros(shape=(spatial_dimension, 1))  # 测量残差
        self.__z_pred = np.zeros(shape=(spatial_dimension, 1))
        self.__S = np.zeros(shape=(spatial_dimension, spatial_dimension))
        self.__H = np.zeros(shape=(spatial_dimension, 2 * spatial_dimension))  # H 命名是否合适
        self.__K = np.zeros(shape=(2 * spatial_dimension, spatial_dimension))

        # 过程噪声标准差
        self.__Q = Q
        # 测量噪声标准差
        self.__R = R

        # F 矩阵初始化
        self.__F = np.eye(2 * spatial_dimension)
        # self.F = np.ones(shape=(2 * num_anchor, 2 * num_anchor))
        for i in range(spatial_dimension):
            self.__F[i, i + spatial_dimension] = delta_t
        # print(self.__F)
        pass

    def set_init_position(self, loc):
        assert loc.shape == (3, ) or loc.shape == (1, 3)
        self.__x[0, 0] = loc[:, 0]
        self.__x[1, 0] = loc[:, 1]
        self.__x[2, 0] = loc[:, 2]
        pass

    def __predict(self):
        """
        状态预测
        Returns:

        """
        # 求x_k,k-1 预测状态
        self.__x = self.__F.dot(self.__x)
        # 求 P_k,k-1 预测估计协方差矩阵
        self.__P = self.__F.dot(self.__P).dot(self.__F.T) + self.__Q

        # 求z_pred，根据预测状态计算观测值
        self.__z_pred = self.__x[0:self.__spatial_dimension]
        # print(self.__z_pred)
        # 求 H
        self.__H = np.zeros(shape=(self.__spatial_dimension, 2 * self.__spatial_dimension))
        self.__H[:, 0:self.__spatial_dimension] = np.eye(self.__spatial_dimension)
        # print(self.__H)
        pass

    def __update(self, location: np.ndarray):
        """
        更新
        Args:
            location: 系统观测，标签位置。维度-（patial_dimension, 1）

        Returns:
            x，状态估计，标签位置，维度-（spatial_dimension，1）, 保存在self.x中
        """
        assert location.shape == (self.__spatial_dimension, 1)
        # 测量残差 y_k
        self.__y = location - self.__z_pred
        # 测量残差协方差 S_k
        self.__S = self.__H.dot(self.__P).dot(self.__H.T) + self.__R
        # 求 K
        self.__K = self.__P.dot(self.__H.T).dot(np.linalg.inv(self.__S))
        # 更新 x
        self.__x = self.__x + self.__K.dot(self.__y)
        self.__P = (np.eye(2 * self.__spatial_dimension) - self.__K.dot(self.__H)).dot(self.__P)
        pass

    def iteration(self, location: np.ndarray) -> np.ndarray:
        """
        迭代，进行扩展卡尔曼滤波。
        Args:
            location: 系统观测，标签位置。维度-（spatial_dimension, 1）

        Returns:
            x，状态估计，标签位置，维度-（spatial_dimension，1）
        """
        assert location.shape == (self.__spatial_dimension, 1)

        self.__predict()
        self.__update(location)
        return self.__x[0:self.__spatial_dimension]
        # return self.__x
    pass
