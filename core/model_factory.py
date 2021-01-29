from enum import Enum

from core.model.stateful_double_lstm_model import generate_stateful_lstm_model, generate_stateful_lstm_model_2


class ModelType(Enum):
    """
    枚举类, 此处用于ModelFactory简单工厂传参使用.
    """
    STATEFUL_DOUBLE_LSTM_MODEL = 1
    STATEFUL_DOUBLE_LSTM_MODEL_2 = 1


class ModelFactory(object):
    """
    神经网络简单模型工厂:
        - 两层lstm网络, "stateful_double_lstm"
    """
    @staticmethod
    def product_model(model_type: ModelType, **kwargs):
        return {
            ModelType.STATEFUL_DOUBLE_LSTM_MODEL: generate_stateful_lstm_model(**kwargs),
            ModelType.STATEFUL_DOUBLE_LSTM_MODEL_2: generate_stateful_lstm_model_2(**kwargs)
        }.get(model_type, None)
