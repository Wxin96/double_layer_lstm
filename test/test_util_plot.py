#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : test_util_plot.py
@ide    : PyCharm
@time   : 21/3/19 21:56
@desc   ：测试画图函数
"""

from unittest import TestCase

from data.data_generator import walk_line_a2b
from util.util_plot import Plot

import numpy as np


class TestUtilPlot(TestCase):

    def test_plot_cdf_chart(self):
        data = [14.27, 14.80, 12.28, 17.09, 15.10, 12.92, 15.56, 15.38,
                15.15, 13.98, 14.90, 15.91, 14.52, 15.63, 13.83, 13.66,
                13.98, 14.47, 14.65, 14.73, 15.18, 14.49, 14.56, 15.03,
                15.40, 14.68, 13.33, 14.41, 14.19, 15.21, 14.75, 14.41,
                14.04, 13.68, 15.31, 14.32, 13.64, 14.77, 14.30, 14.62,
                14.10, 15.47, 13.73, 13.65, 15.02, 14.01, 14.92, 15.47,
                13.75, 14.87, 15.28, 14.43, 13.96, 14.57, 15.49, 15.13,
                14.23, 14.44, 14.57]
        dict_remse = {"data": data}
        dict_color = {"data": "r"}
        Plot.plot_cdf_chart(dict_remse, dict_color)
        pass

    def test_plot_3dtraj_line_chart_test(self):
        point_start = np.array([[-3, -2, 1.4]])
        point_end = np.array([[4, 3, 1.6]])
        traj = walk_line_a2b(point_start, point_end, speed=1.5, delta_t=0.2)
        Plot.plot_3dtraj_line_chart_test(traj)
        pass

    pass
