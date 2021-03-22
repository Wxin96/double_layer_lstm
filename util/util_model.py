#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : util_model.py
@ide    : PyCharm
@time   : 21/3/19 20:21
@desc   ：神经网络模型工具
"""
import numpy as np
from keras.models import load_model
import os


class ModelUtil(object):

    @staticmethod
    def model_predit(model_path: str, ranging_data: np.ndarray):
        """
        模型路径，导入模型， 回归
        :param model_path: 模型的位置
        :param ranging_data: 测距数据，维度-（轨迹长度，4）
        :return: 预测的位置数据， 维度-（1，轨迹长度，3）
        """
        len_traj = len(ranging_data)
        idx = 0    # 前一位
        assert os.path.exists(model_path)  # 检查文件是否存在
        assert ranging_data.shape == (len_traj, 4)

        loc_traj = np.zeros(shape=(len_traj, 3))    # 路径
        model = load_model(model_path)  # 模型
        while idx < len_traj:
            idx_right = min(idx + 30, len_traj)
            ranging_data_single = ranging_data[idx: idx_right].reshape(1, idx_right - idx, 4)
            loc_traj[idx: idx_right] = model.predict(ranging_data_single)[0]
            idx += 30   # 时间步长
            pass

        return loc_traj



    pass
