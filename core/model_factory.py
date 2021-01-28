from enum import Enum

from core.model.stateful_double_lstm_model import generate_stateful_lstm_model


class ModelType(Enum):
    """
    枚举类, 此处用于ModelFactory简单工厂传参使用.
    """
    STATEFUL_DOUBLE_LSTM_MODEL = 1


class ModelFactory(object):
    """
    神经网络简单模型工厂:
        - 两层lstm网络, "stateful_double_lstm"
    """
    @staticmethod
    def product_model(model_type: ModelType, **kwargs):
        return {
            ModelType.STATEFUL_DOUBLE_LSTM_MODEL: generate_stateful_lstm_model(**kwargs)
        }.get(model_type, None)
