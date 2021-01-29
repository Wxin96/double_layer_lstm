#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : stateful_double_lstm_model_exp1.py
@ide    : PyCharm
@time   : 21/1/28 21:53
@desc   ：stateful_double_lstm_model测试
"""
from core.model_factory import ModelFactory, ModelType
from data.data_generator import generator_3d_trajectory, generator_3d_ranging_data
import numpy as np


def test_stateful_double_lstm_mode():
    # 仿真数据
    batch_size = 30
    step_num = 10000
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    los_sd = 20e-3
    nlos_bias = 0.6
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory(step_num=step_num, step_mode=i // 2, length=length, width=width,
                                                high=high,
                                                z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 1)
    print(ranging_batch)
    # model
    model = ModelFactory.product_model(ModelType.STATEFUL_DOUBLE_LSTM_MODEL)
    # TODO: batch_size和time_step不匹配,需要进行调整
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=100)


def test_stateful_double_lstm_mode_2():
    """
    test_stateful_double_lstm_mode()函数的batch_size有问题,重新生成.
    :return: 无返回值
    """
    # 仿真数据
    batch_size = 30
    step_num = 10_020
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    los_sd = 20e-3
    nlos_bias = 0.6
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory(step_num=step_num, step_mode=i // 2, length=length, width=width,
                                                high=high,
                                                z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 1)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.STATEFUL_DOUBLE_LSTM_MODEL_2, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=100)


if __name__ == '__main__':
    test_stateful_double_lstm_mode_2()
    pass

