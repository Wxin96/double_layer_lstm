#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : ekf.py
@ide    : PyCharm
@time   : 2021-02-23 17:14:52
@desc   ：扩展卡尔曼滤波, 测距观测 写程序参考 卡尔曼滤波wiki，https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2
        x_k_k-1 = F_k x_k-1_k-1 + w_k
"""
import numpy as np


class EKF:
    """
    扩展卡尔曼滤波，
        状态，位置、速度
        观测：基站与标签测距
    """

    def __init__(self, num_anchor: int, spatial_dimension: int, anchor_location: np.ndarray, delta_t: float,
                 Q: np.ndarray, R: np.ndarray, P_init: np.ndarray):
        """
        扩展卡尔曼滤波初始化
        Args:
            num_anchor: 基站数目
            spatial_dimension: 测距维数
            anchor_location: 基站坐标，维度-（num_anchor，spatial_dimension）
            delta_t: 间隔时间
            Q: 过程噪声【方差】 维度-（2*spatial_dimension，2*spatial_dimension）
            R: 测量噪声【方差】 维度-（num_anchor，num_anchor）
            P_init: 预测估计协方差矩阵 初始值 维度-（2*spatial_dimension，2*spatial_dimension）
        """
        # 判断
        assert anchor_location.shape == (num_anchor, spatial_dimension)
        assert Q.shape == (2 * spatial_dimension, 2 * spatial_dimension)
        assert R.shape == (num_anchor, num_anchor)
        assert P_init.shape == (2 * spatial_dimension, 2 * spatial_dimension)

        # 基本参数
        self.__num_anchor = num_anchor
        self.__spatial_dimension = spatial_dimension

        # 扩展卡尔曼滤波参数
        self.__x = np.zeros(shape=(2 * spatial_dimension, 1))  # x初始化
        self.__P = P_init
        self.__y = np.zeros(shape=(num_anchor, 1))  # 测量残差
        self.__z_pred = np.zeros(shape=(num_anchor, 1))
        self.__S = np.zeros(shape=(num_anchor, num_anchor))
        self.__H = np.zeros(shape=(num_anchor, 2 * spatial_dimension))  # H 命名是否合适
        self.__K = np.zeros(shape=(2 * spatial_dimension, num_anchor))

        # 基站坐标
        self.__anchor_location = anchor_location

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
        """
        初始化初始位置，维度-（1,3）or（3，）
        :param loc: 初始位置，维度-（1,3）or（3，）
        :return:
        """
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
        x_pred = self.__x[0:self.__spatial_dimension].reshape(3)
        # 求 P_k,k-1 预测估计协方差矩阵
        self.__P = self.__F.dot(self.__P).dot(self.__F.T) + self.__Q

        # 求z_pred，根据预测状态计算观测值
        for i in range(len(self.__z_pred)):
            self.__z_pred[i][0] = np.linalg.norm(x_pred - self.__anchor_location[i])
        # 求 H
        self.__H[:, 0:self.__spatial_dimension] = x_pred - self.__anchor_location
        # print(self.__H)
        # print(self.__z_pred)
        self.__H /= self.__z_pred
        # print(self.__H)
        pass

    def __update(self, ranging_data: np.ndarray):
        """
        更新
        Args:
            ranging_data: 系统观测，基站到标签的测距数据。维度-（num_anchor, 1）

        Returns:
            x，状态估计，标签位置，维度-（spatial_dimension，1）, 保存在self.x中
        """
        assert ranging_data.shape == (self.__num_anchor, 1)
        # 测量残差 y_k
        # self.__y = ranging_data - self.__H.dot(self.__x)
        self.__y = ranging_data - self.__z_pred
        # 测量残差协方差 S_k
        self.__S = self.__H.dot(self.__P).dot(self.__H.T) + self.__R
        # 求 K
        self.__K = self.__P.dot(self.__H.T).dot(np.linalg.inv(self.__S))
        # 更新 x
        self.__x = self.__x + self.__K.dot(self.__y)
        self.__P = (np.eye(2 * self.__spatial_dimension) - self.__K.dot(self.__H)).dot(self.__P)
        pass

    def iteration(self, ranging_data: np.ndarray) -> np.ndarray:
        """
        迭代，进行扩展卡尔曼滤波。
        Args:
            ranging_data: 测距数据，维度-（num_anchor, 1）

        Returns:
            x，状态估计，标签位置，维度-（spatial_dimension，1）
        """
        assert ranging_data.shape == (self.__num_anchor, 1)
        self.__predict()
        self.__update(ranging_data)
        return self.__x[0:self.__spatial_dimension]
        # return self.__x
