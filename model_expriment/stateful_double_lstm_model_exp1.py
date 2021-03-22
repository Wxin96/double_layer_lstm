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
from keras.models import load_model
from core.model_factory import ModelFactory, ModelType
from data.data_generator import generator_3d_trajectory, generator_3d_ranging_data, generator_3d_trajectory_2
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os

from util.util_evaluate import Evaluate
from util.util_model import ModelUtil


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
    # 测试数据生成, batch_size = 1, step_num = 30, 三维坐标
    traj_test = np.zeros(shape=(1, 30, 3))
    traj_test = generator_3d_trajectory(step_num=30, step_mode=1, length=length, width=width, high=high,
                                        z_high=z_high)
    ranging_test, traj_test = generator_3d_ranging_data(traj_test, anchors_loc, origin_coordinate,
                                                        los_sd, nlos_bias, nlos_sd, 1)
    model.save("./save_model/stateful_double_lstm_mode_2_1.h5")
    # traj_predict = model.predict(traj_test, batch_size=1, steps=30)


def test_double_lstm_mode():
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
    nlos_bias = 0.2
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory(step_num=step_num, step_mode=1, length=length, width=width,
                                                high=high,
                                                z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 0)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.DOUBLE_LSTM_MODEL, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=1000)
    # 测试数据生成, batch_size = 1, step_num = 30, 三维坐标
    traj_test = np.zeros(shape=(1, 30, 3))
    traj_test = generator_3d_trajectory(step_num=30, step_mode=1, length=length, width=width, high=high,
                                        z_high=z_high)
    ranging_test, traj_test = generator_3d_ranging_data(traj_test, anchors_loc, origin_coordinate,
                                                        los_sd, nlos_bias, nlos_sd, 0)
    model.save("./save_model/double_lstm_model_epoch_1000_0224.h5")
    traj_predict = model.__predict(traj_batch, batch_size=30)
    print(traj_batch)
    print(traj_predict)
    print(traj_predict - traj_batch)
    # traj_predict = model.predict(traj_test, batch_size=1, steps=30)


def test_double_lstm_mode_line_traj():
    """
    210309测试网络训练直线性能。
    :return: 无返回值
    """
    # 仿真数据
    batch_size = 30
    step_num = 10_020
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))  # 30 10_020 3
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    # los_sd = 20e-3
    los_sd = 0
    nlos_bias = 0.2
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        # traj_batch[i] = generator_3d_trajectory(step_num=step_num, step_mode=1, length=length, width=width,
        #                                         high=high,
        #                                         z_high=z_high)
        traj_batch[i] = generator_3d_trajectory_2(step_num=step_num,
                                                  length=length, width=width, high=high, z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 0)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.DOUBLE_LSTM_MODEL, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=1000)

    model.save("./save_model/double_lstm_model_epoch_1000_0309_line_traj_1.h5")


def test_double_lstm_mode_line_traj_los():
    """
    210311测试网络训练直线性能。加入los误差， 更改了基站z（扩展）
    :return: 无返回值
    """
    # 仿真数据
    batch_size = 30
    step_num = 10_020
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))  # 30 10_020 3
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, -0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, -2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    los_sd = 100e-3
    # los_sd = 0
    nlos_bias = 0.2
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory_2(step_num=step_num,
                                                  length=length, width=width, high=high, z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 0)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.DOUBLE_LSTM_MODEL, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=1000)

    model.save("./save_model/double_lstm_model_epoch_1000_0312_line_traj_los0.1_2.h5")


def test_double_lstm_mode_line_traj_nlos():
    """
    210311测试网络训练直线性能。加入nlos误差， 更改了基站z（扩展）
    :return: 无返回值
    """
    # 仿真数据
    batch_size = 30
    step_num = 10_020
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))  # 30 10_020 3
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3, 1, 0.5], [4, -1., 0.], [3, 3, 2], [-2., 3, 2.5]])
    origin_coordinate = np.array([4, 2, 0])
    los_sd = 50e-3
    # los_sd = 0
    nlos_bias = 0.4
    nlos_sd = 80e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory_2(step_num=step_num,
                                                  length=length, width=width, high=high, z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, mode=1)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.DOUBLE_LSTM_MODEL, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=1000)

    model.save("./save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5")

def test_double_lstm_mode_unit_cm():
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
    los_sd = 30e-3
    nlos_bias = 0.2
    nlos_sd = 45e-3
    ranging_batch = np.zeros(shape=(batch_size, step_num, len(anchors_loc)))
    for i in range(batch_size):
        traj_batch[i] = generator_3d_trajectory(step_num=step_num, step_mode=1, length=length, width=width,
                                                high=high,
                                                z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    0, nlos_bias, nlos_sd, 0)
    ranging_batch = ranging_batch.reshape((step_num // model_time_step * batch_size, model_time_step, len(anchors_loc)))
    traj_batch = traj_batch.reshape((step_num // model_time_step * batch_size, model_time_step, 3))
    ranging_batch = ranging_batch * 100
    traj_batch = traj_batch * 100
    # 当前x-(batch_size, step_num, anchors_loc), y-(batch_size, step_num, 3)
    # 每个x输入维度与基站个数有关
    # model
    model = ModelFactory.product_model(ModelType.DOUBLE_LSTM_MODEL, input_dim=4)
    model.fit(x=ranging_batch, y=traj_batch, batch_size=batch_size, epochs=1000)
    model.save("./save_model/double_lstm_model_epoch_1000_0304.h5")


def test_save_model_1():
    """
    测试 ./save_model/stateful_double_lstm_mode_2_1.h5 中模型, time_step=30, x_dim = 4(基站数目), y_dim = 3(三维坐标,
    在序列中, time_step * y_dim)
    :return:
    """
    # 仿真数据
    batch_size = 30
    # step_num = 1_020
    step_num = 30
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    los_sd = 20e-3
    nlos_bias = 0.2
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
    # model
    model = load_model("./save_model/double_lstm_model_epoch_1000.h5+")
    model.summary()

    traj_predict = model.predict(ranging_batch, batch_size=30, steps=30)
    print(traj_predict)
    print(traj_batch)
    print(traj_predict - traj_batch)


def test_save_model_2():
    """
    测试 ./save_model/stateful_double_lstm_mode_2_1.h5 中模型, time_step=30, x_dim = 4(基站数目), y_dim = 3(三维坐标,
    在序列中, time_step * y_dim)
    :return:
    """
    # 仿真数据
    batch_size = 30
    step_num = 1_020
    model_time_step = 30
    traj_batch = np.zeros(shape=(batch_size, step_num, 3))
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    los_sd = 20e-3
    nlos_bias = 0.2
    nlos_sd = 45e-3
    model = load_model("./save_model/stateful_double_lstm_mode_2_1.h5")
    model.summary()
    # 测试数据生成, batch_size = 1, step_num = 30, 三维坐标
    traj_test = np.zeros(shape=(1, 30, 3))
    traj_test[0] = generator_3d_trajectory(step_num=30, step_mode=1, length=length, width=width, high=high,
                                           z_high=z_high)
    ranging_test = np.zeros(shape=(1, 30, 4))
    ranging_test[0], traj_test[0] = generator_3d_ranging_data(traj_test[0], anchors_loc, origin_coordinate,
                                                              los_sd, nlos_bias, nlos_sd, 1)
    traj_predict = model.__predict(ranging_test, batch_size=1, steps=30)
    # print(ranging_test)
    # print(traj_test)
    # print(traj_predict)


def test_save_model_3():
    """
    测试直线无误差的模型，210310
    :return:
    """
    # 生成数据
    anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    origin_coordinate = np.array([4 * 1.13, 2 * 1.13, 0])
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    # los_sd = 20e-3
    los_sd = 0
    nlos_bias = 0.2
    nlos_sd = 45e-3

    traj_batch = np.zeros(shape=(30, 30, 3))  # batch_size, time_step, output_dim
    ranging_batch = np.zeros(shape=(30, 30, 4))  # batch_size, time_step, input_dim
    for i in range(len(traj_batch)):
        traj_batch[i] = generator_3d_trajectory_2(30, length=length, width=width, high=high, z_high=z_high)
        ranging_batch[i], traj_batch[i] = generator_3d_ranging_data(traj_batch[i], anchors_loc, origin_coordinate,
                                                                    los_sd, nlos_bias, nlos_sd, 0)
        pass
    # 载入模型
    model = load_model("./save_model/double_lstm_model_epoch_1000_0309_line_traj_1.h5")
    model.summary()

    traj_predict = model.predict(ranging_batch)
    print(traj_predict - traj_batch)
    pass


def test_save_model_4():
    """
    测试直线无误差的模型，210311, 去除了stateful
    :return:
    """
    # 生成数据
    anchors_loc = np.array([[-3, 1, 0.5], [4, -1., 0.], [3, 3, 2], [-2., 3, 2.5]])
    origin_coordinate = np.array([4, 2, 0])
    length = 10 * 1.13
    width = 6 * 1.13
    high = 3.5
    z_high = 2.0
    los_sd = 100e-3
    nlos_bias = 0.4
    nlos_sd = 80e-3

    traj_data = generator_3d_trajectory_2(30, length=length, width=width, high=high, z_high=z_high)
    traj_data = np.zeros((60, 3)) + np.array([1, 2, 1.5])
    ranging_data, traj_data = generator_3d_ranging_data(traj_data, anchors_loc, origin_coordinate,
                                                                los_sd, nlos_bias, nlos_sd, mode=0)
    # 载入模型
    model = load_model("./save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5")
    model.summary()
    plot_model(model, to_file='double_lstm_model_0319.png', show_shapes=True, expand_nested=True)
    traj_predict = model.predict(ranging_data.reshape(1, 60, 4))
    print(traj_predict - traj_data)
    print(traj_predict)
    print(
        Evaluate.calc_mean_rmse(np.array([-3, 0, 1.5]), traj_predict.reshape(60, 3))
    )

    # print(ModelUtil.model_predit("../model_expriment/save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5",
    #                              ranging_data))
    pass


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 强制使用CPU
    np.set_printoptions(threshold=np.inf)  # print 打印矩阵完全

    # test_double_lstm_mode_line_traj_nlos()
    test_save_model_4()

    pass
