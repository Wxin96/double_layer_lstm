#!/user/bin/env python
# coding=utf-8
"""
@proj   : double_layer_lstm
@author : Wxin
@file   : traditional_expriment.py
@ide    : PyCharm
@time   : 21/3/16 19:52
@desc   ： 经典定位算法测试
"""
import numpy as np

from data.data_generator import draw_trajectory, generator_3d_ranging_data
from traditional_algorithm.location import Location, LocationType


def fixed_point_los(anchor_loc:np.ndarray):
    """
    视距情况，定点定位。
    测试：Chan、Chan-Taylor、Kalman、C-T-K
    :param anchor_loc: 基站坐标
    :return: 无返回值，作图
    """
    assert anchor_loc.shape == (len(anchor_loc), 3)
    origin_coordinate = np.array([0, 0, 0])
    tag_loc = np.array([[0, 0, 1.8]])
    los_sd = 0.1
    cov_mat = los_sd * los_sd * np.eye(len(anchor_loc))

    print(cov_mat)


    for i in range(1000):
        # 生成测距数据
        ranging_data, tag_loc = generator_3d_ranging_data(tag_loc, anchor_loc, origin_coordinate, los_sd=los_sd, nlos_bias=0, nlos_sd=0, mode=0, nlos_prob=0)

        # chan
        pred_loc_chan_3d = Location.positioning(LocationType.Chan_3d, anchors_loc=anchor_loc, ranging_data=ranging_data.reshape(len(anchor_loc)), cov_mat=cov_mat)
        # chan-taylor
        pred_loc_chan_taylor_3d = Location.positioning(LocationType.Chan_Taylor_3d, anchors_loc=anchor_loc, ranging_data=ranging_data.reshape(len(anchor_loc)), cov_mat=cov_mat)
        # kf

        print(tag_loc)
        print(pred_loc_chan_3d)
        print(pred_loc_chan_taylor_3d)
    pass




if __name__ == '__main__':
    # 房间范围 长： 宽： 高：
    # anchors_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    anchor_loc = np.array([[-3.29, 1.13, 1.66], [3.57, -1.13, 0.925], [3.57, 2.26, 1.950], [-2.26, 3.39, 2.230]])
    fixed_point_los(anchor_loc=anchor_loc)
    pass
