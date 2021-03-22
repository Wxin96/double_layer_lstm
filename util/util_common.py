#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : util_common.py
@ide    : PyCharm
@time   : 21/3/21 9:49
@desc   ：
"""
import numpy as np


class CommonUtil(object):
    @staticmethod
    def nlos_traj_generate(traj: np.ndarray, nlos_record: np.ndarray):
        """
        生成NLOS段的轨迹
        :param traj: 轨迹，（轨迹长度，3）
        :param nlos_record: NLOS标记，（轨迹长度，）
        :return: NLOS轨迹
        """
        assert traj.shape == (len(traj), 3)
        assert nlos_record.shape == (len(traj),)
        traj_nlos = np.array([[0, 0, 0]])
        len_traj = len(traj)
        for i in range(len_traj):
            if nlos_record[i] == 1:
                traj_nlos = np.row_stack((traj_nlos, traj[i]))
        pass
        return traj_nlos[1:-1]
    pass
