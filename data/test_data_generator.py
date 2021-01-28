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
import numpy as np


class Test(TestCase):
    def test_generator_3d_trajectory(self):
        traj = data_generator.generator_3d_trajectory(10, random_loc=True)
        # print(traj)

    def test_draw_trajectory(self):
        traj = data_generator.generator_3d_trajectory(1000, step_mode=1, random_loc=True, z_high=3)
        data_generator.draw_trajectory(traj)

    def test_generator_single_ranging(self):
        anchor_loc = np.array([0, 0, 0])
        tag_loc = np.array([3, 4, 12])
        nlos_dist = data_generator.generator_single_ranging(anchor_loc, tag_loc, 0.5, 45e-3)
        print(nlos_dist)

    def test_generator_3d_ranging_data(self):
        anchors_loc_list = [[0.0, 0.0, 0.0], [0.0, 5.0, 4.0], [4.0, 5.0, 12.0], [10.0, 5.0, 4.0]]
        anchors_loc = np.array(anchors_loc_list)
        origin_coordinate = np.array([2.5, 1.5, 0])
        traj = data_generator.generator_3d_trajectory(10000, step_mode=1, random_loc=True, z_high=3)
        ranging_data,traj = data_generator.generator_3d_ranging_data(traj, anchors_loc, origin_coordinate,
                                                                     20e-3, 0.6, 45e-3, 1, 0.4)
        np.set_printoptions(threshold=np.inf)
        print(ranging_data)
        print(traj)
        print(ranging_data.shape)
        print(traj.shape)
