import numpy as np

from traditional_algorithm.location import Location


class Chan(Location):
    def positioning(self, anchors_loc: np.ndarray, ranging_data: np.ndarray,
                    init_position: np.ndarray = None) -> np.ndarray:
        """
        在三维空间中，利用 基站位置、测距数据、初始位置进行定位。
        :param anchors_loc: 基站位置，维数（num_anchor, 3）
        :param ranging_data: 基站测距数据，维数（num_anchor,）
        :param init_position: 初始估计位置，维数（3,）
        :return: 定位结果, 维数(3, )
        """
        super().param_check(anchors_loc, ranging_data, init_position)
        pass
