#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : test_data_generator.py
@ide    : PyCharm
@time   : 2021-01-26 10:56:55
@desc   ：
"""
from unittest import TestCase
from data import data_generator


class Test(TestCase):
    def test_generator_3d_trajectory(self):
        traj = data_generator.generator_3d_trajectory(10, random_loc=True)
        print(traj)

    def test_draw_trajectory(self):
        traj = data_generator.generator_3d_trajectory(1000, random_loc=True, z_high=2)
        # data_generator.draw_trajectory(traj)



