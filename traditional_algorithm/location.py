from enum import Enum
from abc import ABCMeta, abstractmethod
import numpy as np
import math


class LocationType(Enum):
    """
    枚举类，此处用于Location选择测距方法。
    """
    Chan_3d = 1
    Taylor_3d = 2
    Chan_Taylor_3d = 3


class Location(metaclass=ABCMeta):
    __dimension = 3

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def param_check(anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray,
                    init_position: np.ndarray = None):
        """
        参数检查
        Args:
            anchors_loc: 基站坐标
            ranging_data: 测距数据
            cov_mat: 测距协方差矩阵， 单位：m
            init_position: 初始位置

        Returns:
            无，输入参数有误会抛异常。
        """
        assert anchors_loc.shape == (len(anchors_loc), Location.__dimension)
        assert ranging_data.shape == (len(anchors_loc),)
        assert cov_mat.shape == (len(anchors_loc), len(anchors_loc))
        if init_position is not None:
            assert init_position.shape == (Location.__dimension,)

    @staticmethod
    def positioning(method: LocationType, anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray,
                    init_position: np.ndarray = None, **kwargs) -> np.ndarray or None:
        """
        在三维空间中，利用 基站位置、测距数据、初始位置进行定位。
        :param method: 测距方法
        :param anchors_loc: 基站位置，维数（num_anchor, 3）
        :param ranging_data: 基站测距数据，维数（num_anchor,）
        :param cov_mat: 测距的协方差矩阵, (num_anchor, )，单位：m
        :param init_position: 初始估计位置，维数（3,）
        :return: 定位结果, 维数(3, )
        """
        Location.param_check(anchors_loc, ranging_data, cov_mat, init_position)
        if method is LocationType.Chan_3d:
            return Location.chan_3d(anchors_loc, ranging_data, cov_mat)
        elif method is LocationType.Taylor_3d:
            return Location.taylor_3d(anchors_loc, ranging_data, cov_mat, init_position, **kwargs)
        elif method is LocationType.Chan_Taylor_3d:
            return Location.chan_taylor_3d(anchors_loc, ranging_data, cov_mat)
        else:
            return None
        pass

    @staticmethod
    def chan_3d(anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray):
        anchors_num = len(anchors_loc)
        # 第一次迭代
        # Ga
        Ga = np.ones(shape=(anchors_num, Location.__dimension + 1))
        Ga[:, 0:Location.__dimension] = -2 * anchors_loc
        # H
        H = np.ones(shape=(anchors_num, 1))
        for i in range(anchors_num):
            H[i][0] = ranging_data[i] ** 2 - np.sum(anchors_loc[i] ** 2)
        # Psi
        B = np.diag(ranging_data)  # B
        Q = cov_mat  # Q
        Psi = B.dot(Q).dot(B) * 4
        # Za
        Za, cov_Za = Location.__WLSE(Ga, Psi, H)
        # 第二次迭代
        # Ga
        Ga = np.ones(shape=(Location.__dimension + 1, Location.__dimension))
        Ga[0: Location.__dimension] = np.eye(Location.__dimension)
        # H
        H = Za ** 2
        H[Location.__dimension, 0] = Za[Location.__dimension, 0]

        # Psi
        B[0:Location.__dimension, 0:Location.__dimension] \
            = np.diag(Za.flatten())[0:Location.__dimension, 0:Location.__dimension]  # B
        B[Location.__dimension, Location.__dimension] = 0.5
        Psi = B.dot(cov_Za).dot(B) * 4
        # Za
        Za_2, cov_Za_2 = Location.__WLSE(Ga, Psi, H)
        # 最后处理
        res_Za = Za
        # print(Za)

        # if Za_2.min() > 0:
        #     Za_2 = np.sqrt(Za_2)
        #     for i in range(Location.__dimension):
        #         if Za[i][0] < 0:
        #             Za_2[i][0] *= -1
        #     res_Za = Za_2
        #     print("chan,使用两次迭代")
        # else:
        #     res_Za = Za[0:Location.__dimension]
        #     print("chan,使用一次迭代")

        # print(res_Za)
        res_Za = Za[0:Location.__dimension]
        return res_Za.reshape(3)
        pass

    @staticmethod
    def __WLSE(Ga: np.ndarray, Psi: np.ndarray, H: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Psi = H - Ga * Za
        Args:
            Ga: 系数
            Psi: 偏差
            H: 偏移

        Returns:
            估计的Za, Za的协方差矩阵
        """
        Psi_inv = np.linalg.inv(Psi)
        cov_Za = np.linalg.inv(Ga.T.dot(Psi_inv).dot(Ga))
        Za = cov_Za.dot(Ga.T).dot(Psi_inv).dot(H)
        return Za, cov_Za

    @staticmethod
    def taylor_3d(anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray, init_position: np.ndarray,
                  **kwargs):
        """
        泰勒级数定位（需要初始位置），真实位置-（x，y，z），初始位置 init_position -（x_hat, y_hat, z_hat）
        x = x_hat + delta_x
        y = y_hat + delta_y
        z = z_hat + delta_z
        d_i 真实位置
        d_i_hat 估计位置
        Args:
            anchors_loc: 基站位置，维数（num_anchor, 3）
            ranging_data: 基站测距数据，维数（num_anchor,）
            cov_mat: TOA测距的协方差矩阵, (num_anchor, )，单位：m
            init_position: 初始估计位置，维数（3,）
            kwargs: 其他参数（迭代次数、残差）
        Returns:
            定位结果, 维数(3, )
        """
        # 获取参数
        iteratorNum = kwargs.get('iteratorNum', 500)
        delta = kwargs.get('delta', 0.005)  # delta 默认 5 mm
        num_anchor = len(anchors_loc)

        # 单次taylor定位
        def single_taylor_3d(anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray,
                             init_position: np.ndarray):
            # 1.求真实位置
            pred_dist = np.zeros(shape=(num_anchor, 1))
            for i in range(num_anchor):
                pred_dist[i][0] = np.linalg.norm(anchors_loc[i] - init_position)
            # 2.求G和h
            h_t = ranging_data.reshape((num_anchor, 1)) - pred_dist
            G_t = (init_position - anchors_loc) / pred_dist
            delta_pos, cov_pred_pos = Location.__WLSE(G_t, cov_mat, h_t)
            return delta_pos.reshape(3)

        # 多次Taylor定位迭代
        pos = init_position
        curIterator = 0
        while curIterator < iteratorNum:
            curIterator += 1
            delta_pos = single_taylor_3d(anchors_loc, ranging_data, cov_mat, pos)
            pos = delta_pos + pos
            curDelta = np.linalg.norm(delta_pos)
            if curDelta < delta: break
        print("迭代了", curIterator, "次")
        return pos

    @staticmethod
    def chan_taylor_3d(anchors_loc: np.ndarray, ranging_data: np.ndarray, cov_mat: np.ndarray,
                       **kwargs):
        """
        Chan定位为初始位置的Taylor定位
        Args:
            anchors_loc: 基站位置，维数（num_anchor, 3）
            ranging_data: 基站测距数据，维数（num_anchor,）
            cov_mat: TOA测距的协方差矩阵, (num_anchor, )，单位：m
            kwargs: 其他参数（迭代次数、残差）

        Returns:
            定位结果, 维数(3, )
        """
        init_pos = Location.chan_3d(anchors_loc, ranging_data, cov_mat)
        return Location.taylor_3d(anchors_loc, ranging_data, cov_mat, init_pos, **kwargs)
