#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : util_evaluate.py
@ide    : PyCharm
@time   : 21/3/18 9:36
@desc   ：评价指标通用工具
"""
import numpy as np

class Evaluate(object):

    @staticmethod
    def calc_rmse(pos_true: np.ndarray, pos_pred: np.ndarray):
        """
        返回每个点的RMSE
        :param pos_true: 真实位置，维度-（3，）or（3,1）or（num_traj ， 3）
        :param pos_pred: 预测位置， 维度-（num_traj ， 3）
        :return: 每个点的RMSE（num_traj ， 1）
        """
        assert any((pos_true.shape == (3, ), pos_true.shape == (1, 3), pos_true.shape == pos_pred.shape))
        rmse = np.sqrt(
            ((pos_true - pos_pred) ** 2).mean(axis = 1)
        )
        return rmse

    @staticmethod
    def calc_mean_rmse(pos_true: np.ndarray, pos_pred: np.ndarray):
        """
        返回返回平均RMSE
        :param pos_true: 真实位置，维度-（3，）or（3,1）or（num_traj ， 3）
        :param pos_pred: 预测位置， 维度-（num_traj ， 3）
        :return: 平均 RMSE，维度（1， 1）
        """
        return Evaluate.calc_rmse(pos_true, pos_pred).mean()

    @staticmethod
    def calc_euclidean_dist(pos_true: np.ndarray, pos_pred: np.ndarray):
        """
        返回每个点的RMSE
        :param pos_true: 真实位置，维度-（3，）or（3,1）or（num_traj ， 3）
        :param pos_pred: 预测位置， 维度-（num_traj ， 3）
        :return: 每个点的欧几里得距离（num_traj ， 1）
        """
        assert any((pos_true.shape == (3, ), pos_true.shape == (1, 3), pos_true.shape == pos_pred.shape))
        euclidean_dist = np.sqrt(
            ((pos_true - pos_pred) ** 2).sum(axis=1)
        )
        return euclidean_dist

    @staticmethod
    def calc_mean_euclidean_dist(pos_true: np.ndarray, pos_pred: np.ndarray):
        """
        返回每个点的RMSE
        :param pos_true: 真实位置，维度-（3，）or（3,1）or（num_traj ， 3）
        :param pos_pred: 预测位置， 维度-（num_traj ， 3）
        :return: 每个点的欧几里得距离（num_traj ， 1）
        """

        return Evaluate.calc_euclidean_dist(pos_true, pos_pred).mean()

    pass
