#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : thesis_expriment.py
@ide    : PyCharm
@time   : 21/3/16 19:52
@desc   ： 经典定位算法测试
"""
import numpy as np

from data.data_generator import draw_trajectory, generator_3d_ranging_data, walk_line_a2b
from macro.Mode import RangingMode
from traditional_algorithm.ekf import EKF
from traditional_algorithm.kf import KF
from traditional_algorithm.location import Location, LocationType
from util.util_common import CommonUtil
from util.util_evaluate import Evaluate
from util.util_model import ModelUtil
from util.util_plot import Plot


def fixed_point_los(anchor_loc: np.ndarray, itera_num: int = 500, los_sd: float = 0.01, plot_flag: bool = True):
    """
    视距情况，定点定位。
    测试：Chan、Chan-Taylor、Kalman、C-T-K
    :param anchor_loc: 基站坐标
    :param los_sd: 测距标准差
    :param itera_num: 迭代次数
    :param plot_flag: 是否画图
    :return: 无返回值，作图
    """
    assert anchor_loc.shape == (len(anchor_loc), 3)
    origin_coordinate = np.array([0, 0, 0])
    tag_loc = np.array([[0, 0, 1]])  # 0 0 1.2
    # tag_loc = np.array([[0, 0, 1.2]])     # 0 0 1.2
    # los_sd = 0.05
    nlos_bias = 0.4
    nlos_sd = 0.08
    cov_mat = los_sd * los_sd * np.eye(len(anchor_loc))

    # 扩展卡尔曼参数
    spatial_dimension = 3
    num_anchor = 4
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = cov_mat
    P_init = np.eye(2 * spatial_dimension)
    ekf = EKF(num_anchor, spatial_dimension, anchor_loc, 0.2, Q, R, P_init)

    # 卡尔曼初始化
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = los_sd * los_sd * np.eye(3)
    P_init = np.eye(2 * spatial_dimension)

    kf = KF(spatial_dimension, 0.2, Q, R, P_init)

    # 数据统计，次数
    # itera_num = 510
    pred_loc_chan_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_chan_taylor_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_ekf_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_c_t_k_3d = np.zeros(shape=(itera_num, 3))

    # 轨迹
    traj = np.zeros(shape=(itera_num, 3)) + tag_loc

    ranging_data, tag_loc = generator_3d_ranging_data(traj, anchor_loc, origin_coordinate, los_sd=los_sd,
                                                      nlos_bias=nlos_bias, nlos_sd=nlos_sd, mode=1, nlos_prob=0)
    for i in range(itera_num):
        # 生成测距数据
        # chan
        pred_loc_chan_3d[i] = Location.positioning(LocationType.Chan_3d, anchors_loc=anchor_loc,
                                                   ranging_data=ranging_data[i], cov_mat=cov_mat)
        # chan-taylor
        pred_loc_chan_taylor_3d[i] = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc=anchor_loc,
                                                          ranging_data=ranging_data[i], cov_mat=cov_mat)
        # ekf
        pred_loc_ekf_3d[i] = Location.positioning(LocationType.EKF_3d, anchor_loc,
                                                  ranging_data[i], cov_mat, ekf=ekf)
        # kf
        pred_loc_c_t_k_3d[i] = Location.positioning(LocationType.C_T_K_3d, anchor_loc,
                                                    ranging_data[i], cov_mat, kf=kf)
    # 神经网络预测
    pred_loc_lstm = ModelUtil.model_predit(
        "../model_expriment/save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5",
        ranging_data)
    # 计算rmse
    rmse_chan = Evaluate.calc_mean_rmse(tag_loc, pred_loc_chan_3d)
    rmse_c_t = Evaluate.calc_mean_rmse(tag_loc, pred_loc_chan_taylor_3d)
    rmse_ekf = Evaluate.calc_mean_rmse(tag_loc, pred_loc_ekf_3d)
    rmse_c_t_k = Evaluate.calc_mean_rmse(tag_loc, pred_loc_c_t_k_3d)
    rmse_lstm = Evaluate.calc_mean_rmse(tag_loc, pred_loc_lstm)

    if plot_flag:
        dict = {"Chan": pred_loc_chan_3d, "C-T": pred_loc_chan_taylor_3d, "EKF": pred_loc_ekf_3d,
                "C-T-K": pred_loc_c_t_k_3d, "pos_true": tag_loc}
        color_dict = {"Chan": "k", "C-T": "b", "EKF": "r", "C-T-K": "g", "pos_true": "y"}
        Plot.plot_scatter_3d(dict, color_dict)

    evaluate_dict = {"Chan": rmse_chan, "C-T": rmse_c_t, "EKF": rmse_ekf, "C-T-K": rmse_c_t_k, "LSTM": rmse_lstm}
    return evaluate_dict
    pass


def fixed_point_nlos(anchor_loc: np.ndarray, itera_num: int = 500, los_sd: float = 0.05):
    """
    非视距情况，定点定位。
    测试：Chan、Chan-Taylor、Kalman、C-T-K
    :param anchor_loc: 基站坐标
    :param los_sd: 测距标准差
    :param itera_num: 迭代次数
    :param plot_flag: 是否画图
    :return: 无返回值，作图
    """
    assert anchor_loc.shape == (len(anchor_loc), 3)
    origin_coordinate = np.array([0, 0, 0])
    tag_loc = np.array([[0, 0, 1.2]])  # 0 0 1.2
    # tag_loc = np.array([[0, 0, 1.2]])     # 0 0 1.2
    # los_sd = 0.05
    nlos_bias = 0.3
    nlos_sd = 0.08
    cov_mat = los_sd * los_sd * np.eye(len(anchor_loc))

    # 扩展卡尔曼参数
    spatial_dimension = 3
    num_anchor = 4
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = cov_mat
    P_init = np.eye(2 * spatial_dimension)
    ekf = EKF(num_anchor, spatial_dimension, anchor_loc, 0.2, Q, R, P_init)

    # 卡尔曼初始化
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = np.diag(np.array([0.0025, 0.0025, 0.0025]))
    P_init = np.eye(2 * spatial_dimension)

    kf = KF(spatial_dimension, 0.2, Q, R, P_init)

    # 数据统计，次数
    # itera_num = 510
    pred_loc_chan_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_chan_taylor_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_ekf_3d = np.zeros(shape=(itera_num, 3))
    pred_loc_c_t_k_3d = np.zeros(shape=(itera_num, 3))

    # 轨迹
    traj = np.zeros(shape=(itera_num, 3)) + tag_loc

    ranging_data, tag_loc = generator_3d_ranging_data(traj, anchor_loc, origin_coordinate, los_sd=los_sd,
                                                      nlos_bias=nlos_bias, nlos_sd=nlos_sd, mode=1, nlos_prob=1)
    for i in range(itera_num):
        # 生成测距数据
        # chan
        pred_loc_chan_3d[i] = Location.positioning(LocationType.Chan_3d, anchors_loc=anchor_loc,
                                                   ranging_data=ranging_data[i], cov_mat=cov_mat)
        # chan-taylor
        pred_loc_chan_taylor_3d[i] = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc=anchor_loc,
                                                          ranging_data=ranging_data[i], cov_mat=cov_mat)
        # ekf
        pred_loc_ekf_3d[i] = Location.positioning(LocationType.EKF_3d, anchor_loc,
                                                  ranging_data[i], cov_mat, ekf=ekf)
        # kf
        pred_loc_c_t_k_3d[i] = Location.positioning(LocationType.C_T_K_3d, anchor_loc,
                                                    ranging_data[i], cov_mat, kf=kf)
    # 神经网络预测
    pred_loc_lstm = ModelUtil.model_predit(
        "../model_expriment/save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5",
        ranging_data)
    # 计算rmse
    rmse_chan = Evaluate.calc_rmse(tag_loc, pred_loc_chan_3d)
    rmse_c_t = Evaluate.calc_rmse(tag_loc, pred_loc_chan_taylor_3d)
    rmse_ekf = Evaluate.calc_rmse(tag_loc, pred_loc_ekf_3d)
    rmse_c_t_k = Evaluate.calc_rmse(tag_loc, pred_loc_c_t_k_3d)
    rmse_lstm = Evaluate.calc_rmse(tag_loc, pred_loc_lstm)

    dict_rmse = {"Chan": rmse_chan, "C-T": rmse_c_t, "EKF": rmse_ekf, "C-T-K": rmse_c_t_k, "LSTM": rmse_lstm}
    color_dict = {"Chan": "c", "C-T": "m", "EKF": "r", "C-T-K": "g", "LSTM": "b", "pos_true": "y"}

    Plot.plot_cdf_chart(dict_rmse, color_dict)
    pass


def diff_sd_line_chart(nlos_list: list, anchor_loc: np.ndarray):
    """
    不同的
    :param nlos_list: 非视距误差列表
    :param anchor_loc: 基站坐标
    :return:
    """
    num_los = len(nlos_list)
    rmse_c = np.zeros(shape=(num_los, 2))
    rmse_c_t = np.zeros(shape=(num_los, 2))
    rmse_ekf = np.zeros(shape=(num_los, 2))
    rmse_c_t_k = np.zeros(shape=(num_los, 2))
    rmse_lstm = np.zeros(shape=(num_los, 2))

    # 迭代
    for i in range(num_los):
        rmse_c[i, 0] = rmse_c_t[i, 0] = rmse_ekf[i, 0] = rmse_c_t_k[i, 0] = rmse_lstm[i, 0] = nlos_list[i]
        evaluate_dict = fixed_point_los(anchor_loc, los_sd=nlos_list[i], plot_flag=False)
        rmse_c[i, 1] = evaluate_dict.get("Chan")
        rmse_c_t[i, 1] = evaluate_dict.get("C-T")
        rmse_ekf[i, 1] = evaluate_dict.get("EKF")
        rmse_c_t_k[i, 1] = evaluate_dict.get("C-T-K")
        rmse_lstm[i, 1] = evaluate_dict.get("LSTM")
        pass
    # print(rmse_c)
    los_dict = {"Chan": rmse_c, "C-T": rmse_c_t, "EKF": rmse_ekf, "C-T-K": rmse_c_t_k, "LSTM": rmse_lstm}
    color_dict = {"Chan": "c", "C-T": "m", "EKF": "r", "C-T-K": "g", "LSTM": "b", "pos_true": "y"}
    Plot.plot_line_chart(los_dict, color_dict)
    pass


def move_point(anchor_loc: np.ndarray, point_start: np.ndarray, point_end: np.ndarray, mode: RangingMode, **kwargs):
    """
    移动定位，有两种模式，LOS NLOs
    :param anchor_loc: 基站坐标，维度-（基站个数，3）
    :param point_start: 起始位置，维度（1，3）
    :param point_end: 终点位置，维度（1,3）
    :param mode: RangingMode.NLOS, RangingMode.LOS
    :param kwargs: 配置LOS，和NLOS参数
    :return:
    """
    assert anchor_loc.shape == (len(anchor_loc), 3)
    assert point_start.shape == (1, 3)
    assert point_end.shape == (1, 3)
    # 获取参数
    los_sd = kwargs.get('los_sd', 0.04)
    nlos_sd = kwargs.get("nlos_sd", 0.08)
    nlos_bias = kwargs.get("nlos_bias", 0.4)
    nlos_prob = kwargs.get("nlos_prob", 0.4)
    # 模式
    if mode == RangingMode.LOS:
        print("LOS Mode, los_sd = " + str(los_sd))
    elif mode == RangingMode.NLOS:
        print("NLOS Mode, los_sd = " + str(los_sd) + ", nlos_prob = " + str(nlos_prob)
              + ", nlos_sd = " + str(nlos_sd) + ", nlos_bias = " + str(nlos_bias))
    # 获取轨迹
    traj = walk_line_a2b(point_start, point_end, speed=1.5, delta_t=0.2)
    len_traj = len(traj)
    ranging_data, traj, nlos_record = generator_3d_ranging_data(traj, anchor_loc, np.array([0, 0, 0]), los_sd,
                                                                nlos_bias, nlos_sd, mode, nlos_prob)
    traj_nlos = CommonUtil.nlos_traj_generate(traj, nlos_record)

    # 神经网络预测
    pred_traj_lstm = ModelUtil.model_predit(
        "../model_expriment/save_model/double_lstm_model_epoch_1000_0318_line_traj_nlos_3.h5",
        ranging_data)

    # 参数
    cov_mat = los_sd * los_sd * np.eye(len(anchor_loc))
    # 扩展卡尔曼参数
    spatial_dimension = 3
    num_anchor = 4
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = los_sd * los_sd * np.eye(len(anchor_loc))
    P_init = np.eye(2 * spatial_dimension)
    ekf = EKF(num_anchor, spatial_dimension, anchor_loc, 0.2, Q, R, P_init)
    # ekf.set_init_position(point_start)

    # 卡尔曼初始化
    Q = np.diag(np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))
    R = los_sd * los_sd * np.eye(3)
    P_init = np.eye(2 * spatial_dimension)
    kf = KF(spatial_dimension, 0.2, Q, R, P_init)
    # kf.set_init_position(point_start)

    # 轨迹
    pred_traj_ekf_3d = np.zeros(shape=(len_traj, 3))
    pred_traj_c_t_k_3d = np.zeros(shape=(len_traj, 3))

    for i in range(len_traj):
        # ekf
        pred_traj_ekf_3d[i] = Location.positioning(LocationType.EKF_3d, anchor_loc,
                                                   ranging_data[i], cov_mat, ekf=ekf)
        # kf
        pred_traj_c_t_k_3d[i] = Location.positioning(LocationType.C_T_K_3d, anchor_loc,
                                                     ranging_data[i], cov_mat, kf=kf)
        pass

    dict_traj = {"true_traj": traj, "EKF": pred_traj_ekf_3d, "C-T-K": pred_traj_c_t_k_3d, "LSTM": pred_traj_lstm, "nlos_mark": traj_nlos}
    dict_color = {"true_traj": "k", "EKF": "r", "C-T-K": "g", "LSTM": "b", "nlos_mark": "y"}
    Plot.plot_3dtraj_line_chart(dict_traj, dict_color, mode=mode)

    # 误差
    euclid_error_lstm = np.zeros(shape=(len_traj, 2))
    euclid_error_ekf = np.zeros(shape=(len_traj, 2))
    euclid_error_c_t_k = np.zeros(shape=(len_traj, 2))
    euclid_error_lstm[:, 0] = euclid_error_ekf[:, 0] = euclid_error_c_t_k[:, 0] = np.array(range(0, len_traj)) * 0.2
    euclid_error_lstm[:, 1] = Evaluate.calc_rmse(traj, pred_traj_lstm)
    euclid_error_ekf[:, 1] = Evaluate.calc_rmse(traj, pred_traj_ekf_3d)
    euclid_error_c_t_k[:, 1] = Evaluate.calc_rmse(traj, pred_traj_c_t_k_3d)

    dict_dist = {"EKF": euclid_error_ekf, "C-T-K": euclid_error_c_t_k, "LSTM": euclid_error_lstm}
    dict_color = {"EKF": "r", "C-T-K": "g", "LSTM": "b"}
    Plot.plot_rmse_error_line_chart(dict_dist, dict_color, mode)

    print(Evaluate.calc_mean_rmse(traj, pred_traj_lstm))
    print(Evaluate.calc_mean_rmse(traj, pred_traj_ekf_3d))
    print(Evaluate.calc_mean_rmse(traj, pred_traj_c_t_k_3d))

    pass


if __name__ == '__main__':
    # 房间范围 长： 宽： 高：
    # anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    anchor_loc = np.array([[-3, 1, 0.5], [4, -1., 0.], [3, 3, 2], [-2., 3, 2.5]])
    # evaluate_dict = fixed_point_los(anchor_loc=anchor_loc, los_sd=0.05)
    # print(evaluate_dict)
    # los_list = [0.01, 0.05, 0.1, 0.2]
    # diff_sd_line_chart(los_list, anchor_loc)
    # fixed_point_nlos(anchor_loc, itera_num=510, los_sd=0.05)

    # 移动标签测试
    point_start = np.array([[-3, -2, 0.2]])
    point_end = np.array([[4, 3, 1.6]])
    move_point(anchor_loc, point_start=point_start, point_end=point_end, mode=RangingMode.NLOS)
    pass
