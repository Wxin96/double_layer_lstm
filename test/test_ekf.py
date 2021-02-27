#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Apollo
@file   : test_ekf.py
@ide    : PyCharm
@time   : 2021-02-23 17:58:20
@desc   ：
"""
from unittest import TestCase

from data.data_generator import generate_point_location_ranging
from traditional_algorithm.ekf import EKF
import numpy as np

from traditional_algorithm.kf import KF
from traditional_algorithm.location import Location, LocationType


class TestEKF(TestCase):
    def test_ekf(self):
        # anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        anchors_loc = np.array([[-3.29, 1.13, -1.66], [3.57, -1.13, 0.925], [3.57, 2.26, -1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([-1.32, 0.64, 1.43])
        cov_mat = np.diag(np.array([0.0003, 0.0003, 0.0003, 0.0003]))  # 测距协方差
        # point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0, nlos_bias=0.2,
        #                                                 nlos_sd=0.05, los_sd=0.0173)
        spatial_dimension = 3
        num_anchor = 4
        Q = np.diag(np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))
        R = cov_mat
        P_init = np.eye(2 * spatial_dimension)

        ekf = EKF(num_anchor, spatial_dimension, anchors_loc, 0.2, Q, R, P_init)
        for i in range(20):
            # point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.0, nlos_bias=0.2,
            #                                                 nlos_sd=0.05, los_sd=0.0173)
            point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.0, nlos_bias=0.2,
                                                            nlos_sd=0.05, los_sd=0.0173)
            pos = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc, point_ranging,
                                       cov_mat, iteratorNum=501, delta=0.005)
            print('pos: ', pos)
            loc = ekf.iteration(point_ranging.reshape((4, 1)))
            print(loc, '\n')
        pass

    def test_kf(self):
        # anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
        anchors_loc = np.array([[-3.29, 1.13, -1.66], [3.57, -1.13, 0.925], [3.57, 2.26, -1.950], [-2.26, 3.39, 2.230]])
        tag_loc = np.array([-1.32, 0.64, 1.43])
        cov_mat = np.diag(np.array([0.0003, 0.0003, 0.0003, 0.0003]))  # 测距协方差
        # point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0, nlos_bias=0.2,
        #                                                 nlos_sd=0.05, los_sd=0.0173)
        spatial_dimension = 3
        num_anchor = 4
        Q = np.diag(np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]))
        R = np.diag(np.array([0.001, 0.001, 0.001]))
        P_init = np.eye(2 * spatial_dimension)

        kf = KF(spatial_dimension, 0.2, Q, R, P_init)
        for i in range(1000):
            # point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.0, nlos_bias=0.2,
            #                                                 nlos_sd=0.05, los_sd=0.0173)
            point_ranging = generate_point_location_ranging(anchors_loc, tag_loc, nlos_prob=0.0, nlos_bias=0.2,
                                                            nlos_sd=0.05, los_sd=0.0173)
            pos = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc, point_ranging,
                                       cov_mat, iteratorNum=501, delta=0.005)
            print('pos: ', pos)
            loc = kf.iteration(pos.reshape((spatial_dimension, 1)))
            print(loc, '\n')
        pass
        pass
    pass
