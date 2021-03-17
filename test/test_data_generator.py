#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : test_data_generator.py
@ide    : PyCharm
@time   : 2021-01-26 10:56:55
@desc   ：测试 test_data_generator 文件中函数
"""
from unittest import TestCase
from data import data_generator
import numpy as np

from data.data_generator import generate_point_location_ranging, walk_line_a2b, draw_trajectory, \
    generator_3d_trajectory_2


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
        ranging_data, traj = data_generator.generator_3d_ranging_data(traj, anchors_loc, origin_coordinate,
                                                                      20e-3, 0.6, 45e-3, 1, 0.4)
        np.set_printoptions(threshold=np.inf)
        print(ranging_data)
        print(traj)
        print(ranging_data.shape)
        print(traj.shape)

    def test_light_num(self):
        light_num(540, 16, [0, 90, 180, 270, 360, 450, 540])
        print()
        light_num(270, 16, [0, 45, 90,  135, 180, 225, 270])

    def test_generate_point_location_ranging(self):
        anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([0, 1.13, 1.0])
        point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, 0, 0)
        print(point_ranging)

    def test_walk_line_a2b(self):
        a_loc = np.array([[0, 0, 0]])
        b_loc = np.array([[3, 4, 13]])
        speed = 1.5     # unit: m/s
        delta_t = 0.2   # unit: s
        traj = walk_line_a2b(a_loc, b_loc, speed, delta_t)
        pass

    def test_generator_3d_trajectory_2(self):
        traj = generator_3d_trajectory_2(1000)
        # print(traj)
        # draw_trajectory(traj)
        pass


def light_num(range: int, digital_num: int, list: []):
    """
    舞台灯角度测试
    :param range:
    :param digital_num:
    :param list:
    :return:
    """
    for angle in list:
        num = 2 ** digital_num * angle // range
        low = num & 0xff
        high = (num >> 8) & 0xff
        print(angle, high, low)