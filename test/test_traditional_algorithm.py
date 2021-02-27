#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : test_traditional_algorithm.py
@ide    : PyCharm
@time   : 2021-02-07 11:05:56
@desc   ：
"""
from unittest import TestCase
import numpy as np

from data.data_generator import generate_point_location_ranging
from traditional_algorithm.location import Location, LocationType


class TestTraditionalAlgorithm(TestCase):
    def test_positioning(self):
        anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([-1.13, 1.13, 1.0])
        cov_mat = np.diag(np.array([0.0003, 0.0003, 0.0003, 0.0003]))
        point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.1, nlos_bias=0.1,
                                                        nlos_sd=0.05, los_sd=0.017)
        Location.positioning(LocationType.Chan_3d, anchors_loc, point_ranging, cov_mat)

    def test_taylor_positioning(self):
        anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([-1.13, 1.13, 1.0])
        cov_mat = np.diag(np.array([0.0003, 0.0003, 0.0003, 0.0003]))
        point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.2, nlos_bias=0.2,
                                                        nlos_sd=0.05, los_sd=0.017)
        init_pos = Location.positioning(LocationType.Chan_3d, anchors_loc, point_ranging, cov_mat)
        print('init_pos: ', init_pos)
        pos = Location.positioning(LocationType.Taylor_3d, anchors_loc, point_ranging, cov_mat,
                                   init_pos.reshape(3), iteratorNum=501, delta=0.005)
        print('pos: ', pos)
        pass

    def test_chan_taylor_3d(self):
        anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([-1.13, 1.13, 1.0])
        cov_mat = np.diag(np.array([0.0003, 0.0003, 0.0003, 0.0003]))
        point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.2, nlos_bias=0.2,
                                                        nlos_sd=0.05, los_sd=0.017)
        pos = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc, point_ranging,
                                   cov_mat, iteratorNum=501, delta=0.005)
        print('pos: ', pos)
        print('error: ', np.linalg.norm(tag_loc - pos))