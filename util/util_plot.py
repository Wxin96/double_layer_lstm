#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : util_plot.py
@ide    : PyCharm
@time   : 21/3/18 10:22
@desc   ：作图
"""
import mpl_toolkits.mplot3d as p3d
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm  # recommended import according to the docs

from macro.Mode import RangingMode


class Plot(object):

    @staticmethod
    def plot_scatter_3d_test(pos_1: np):
        x = pos_1[:, 0]
        y = pos_1[:, 1]
        z = pos_1[:, 2]
        fig = plt.figure()
        ax = p3d.Axes3D(fig)

        ax.scatter(x, y, z, c='b', s=1, alpha=0.5, label='chan', marker='d')
        ax.legend(['chan', 'taylor'])

        fig.show()
        pass

    @staticmethod
    def plot_scatter_3d(pos_dict: dict, color_dict: dict):
        """
        定点散点图
        :param pos_dict: 方法（字符串）：预测坐标（矩阵， 维度-（num_traj，3））
        :param color_dict: 不同坐标不同颜色
        :return:
        """
        fig = plt.figure()
        ax = p3d.Axes3D(fig)
        s = 1
        alpha = 0.1
        for key in pos_dict:
            loc = pos_dict.get(key)
            x = loc[:, 0]
            y = loc[:, 1]
            z = loc[:, 2]
            # print(key)
            if key == "pos_true":
                ax.scatter(x, y, z, c=color_dict.get(key), s=300, alpha=1, marker='*')
            else:
                ax.scatter(x, y, z, c=color_dict.get(key), s=s, alpha=0.8, marker='.')
            s += 1
            alpha += 0.1
            pass

        # 标注
        ax.legend(pos_dict.keys())

        # x轴标注
        ax.set_xlabel("x/m")

        # y轴标注
        ax.set_ylabel("y/m")

        # z轴标注
        ax.set_zlabel("z/m")

        plt.savefig("./location_scatter.png")

        fig.show()
        pass

    pass

    @staticmethod
    def plot_line_chart_test():
        x = [1, 2, 3]
        y = x
        plt.plot(x, y)
        plt.show()
        pass

    @staticmethod
    def plot_line_chart(los_dict: dict, color_dict: dict):
        """
        画折线图
        :param los_dict: los字典
        :param color_dict:  颜色字典
        :return:
        """
        for key in los_dict:
            los_data = los_dict.get(key)
            x = los_data[:, 0]
            y = los_data[:, 1]
            plt.plot(x, y, color=color_dict.get(key))
            pass
        # x轴标注
        plt.xlabel("SD/m")
        # y轴标注
        plt.ylabel("RMSE/m")

        plt.legend(color_dict.keys())
        plt.savefig("./location_5sd_line.png")
        plt.show()
        pass

    @staticmethod
    def plot_cdf_chart_test():
        data = [14.27, 14.80, 12.28, 17.09, 15.10, 12.92, 15.56, 15.38,
                15.15, 13.98, 14.90, 15.91, 14.52, 15.63, 13.83, 13.66,
                13.98, 14.47, 14.65, 14.73, 15.18, 14.49, 14.56, 15.03,
                15.40, 14.68, 13.33, 14.41, 14.19, 15.21, 14.75, 14.41,
                14.04, 13.68, 15.31, 14.32, 13.64, 14.77, 14.30, 14.62,
                14.10, 15.47, 13.73, 13.65, 15.02, 14.01, 14.92, 15.47,
                13.75, 14.87, 15.28, 14.43, 13.96, 14.57, 15.49, 15.13,
                14.23, 14.44, 14.57]
        # =============绘制cdf图===============
        ecdf = sm.distributions.ECDF(data)
        # 等差数列，用于绘制X轴数据
        x = np.linspace(min(data), max(data))
        # x轴数据上值对应的累计密度概率
        y = ecdf(x)
        # 绘制阶梯图
        plt.step(x, y)
        plt.show()
        # ===============绘制条形图=============
        fig, ax0 = plt.subplots(nrows=1, figsize=(6, 6))
        # 第二个参数是柱子宽一些还是窄一些，越大越窄越密
        ax0.hist(data, 10, density=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
        pass

    @staticmethod
    def plot_cdf_chart(dict_rmse: dict, dict_color: dict):
        """
        根据数据字典和颜色字典绘制CDF图
        :param dict_rmse: REMSE字典
        :param dict_color: 颜色字典
        :return:
        """

        for key in dict_rmse:
            data = dict_rmse.get(key)
            # =============绘制cdf图===============
            ecdf = sm.distributions.ECDF(data)
            # 等差数列，用于绘制X轴数据
            x = np.linspace(min(data), max(data))
            # x轴数据上值对应的累计密度概率
            y = ecdf(x)
            # 绘制阶梯图
            # plt.step(x, y, c=dict_color.get(key))
            plt.plot(x, y, c=dict_color.get(key))
            pass

        plt.legend(dict_rmse.keys())
        plt.xlabel("RMSE/m")
        plt.ylabel("CDF")
        plt.savefig("./location_nlos_cdf.png")
        plt.show()

        pass

    @staticmethod
    def plot_3dtraj_line_chart_test(traj: np.ndarray):
        assert traj.shape == (len(traj), 3)
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]

        ax = plt.subplot(111, projection='3d')
        ax.plot(x, y, z, c='r')
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')

        plt.show()
        pass

    @staticmethod
    def plot_3dtraj_line_chart(dict_traj: dict, dict_color: dict, mode: RangingMode):
        """
        绘制三维折线图
        :param dict_traj: 轨迹字典，单个轨迹维度-（轨迹长度，3）
        :param dict_color: 轨迹颜色字典
        :param mode: RangingMode.NLOS RangingMode.LOS
        :return:
        """
        if mode == RangingMode.NLOS:
            dict_traj.pop("EKF")
        ax = plt.subplot(111, projection='3d')
        for key in dict_traj:
            traj = dict_traj.get(key)
            x = traj[:, 0]
            y = traj[:, 1]
            z = traj[:, 2]
            ax.plot(x, y, z, c=dict_color.get(key))
            pass

        ax.set_zlabel('Z/m')  # 坐标轴

        ax.set_ylabel('Y/m')
        ax.set_xlabel('X/m')
        ax.legend(dict_traj.keys())

        if mode == RangingMode.LOS:
            plt.savefig("./3dtraj_line_chart_los.png")
            pass
        elif mode == RangingMode.NLOS:
            plt.savefig("./3dtraj_line_chart_nlos.png")
            pass
        # ax.set_zlim(0, 1.8)
        plt.show()
        pass


    @staticmethod
    def plot_rmse_error_line_chart(dict_dist: dict, dict_color: dict, mode: RangingMode):
        """
        绘制三维折线图
        :param dict_dist: 误差字典，单个轨迹维度-（轨迹长度）
        :param dict_color: 轨迹颜色字典
        :param mode: RangingMode.NLOS RangingMode.LOS
        :return:
        """
        if mode == RangingMode.NLOS:
            dict_dist.pop("EKF")
            pass
        for key in dict_dist:
            traj = dict_dist.get(key)
            x = traj[:, 0]
            y = traj[:, 1]
            plt.plot(x, y,c=dict_color.get(key))
            pass

        plt.ylabel('RMSE/m')
        plt.xlabel('Time/s')
        plt.legend(dict_dist.keys())
        # ax.set_zlim(0, 1.8)

        if mode == RangingMode.LOS:
            plt.savefig("./euclid_error_line_chart_los.png")
            pass
        elif mode == RangingMode.NLOS:
            plt.savefig("./euclid_error_line_chart_nlos.png")
        plt.show()
        pass
