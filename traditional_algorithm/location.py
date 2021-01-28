from abc import ABCMeta, abstractmethod
import numpy as np


class Location(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def positioning(self, anchors_loc: np.ndarray, ranging_data: np.ndarray,
                    init_position: np.ndarray = None) -> np.ndarray:
        """
        在三维空间中，利用 基站位置、测距数据、初始位置进行定位。
        :param anchors_loc: 基站位置，维数（num_anchor, 3）
        :param ranging_data: 基站测距数据，维数（num_anchor,）
        :param init_position: 初始估计位置，维数（3,）
        :return: 定位结果, 维数(3, )
        """
        pass

    def param_check(self, anchors_loc: np.ndarray, ranging_data: np.ndarray,
                    init_position: np.ndarray = None) -> bool:
        assert anchors_loc.shape == (len(anchors_loc), 3)
        assert ranging_data.shape == (len(anchors_loc),)
        assert init_position.shape == (3,)
        return True
