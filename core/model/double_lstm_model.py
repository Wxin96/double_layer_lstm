from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Bidirectional
import numpy as np
from keras.utils import plot_model


def generate_lstm_model(**kwargs) -> Sequential:
    print("generate_lstm_model")
    batch_size = kwargs.get('batch_size', 30)  # 训练样本批次
    time_step = kwargs.get('time_step', 30)  # 时间步长
    input_dim = kwargs.get('input_dim', 4)  # x的特征数目
    hidden_units = kwargs.get('hidden_units', 256)  # 隐藏层神经元个数
    dropout_rate = kwargs.get('dropout_rate', 0.2)  # 正则化层概率

    # 训练与预测一致，batch_size = None
    batch_size = None

    # 期望输入数据尺寸: (batch_size, time_step, input_dim)
    # 请注意，我们必须提供完整的 batch_input_shape, 因为网络是有状态的。
    # 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
    # 关于 k 批的思考，多段路径对应批中的路径，不会发生混乱
    model = Sequential()
    # 个人猜测 recurrent_activation为门激活函数, activation为输入 c~的激活函数

    # 一层
    model.add(
        LSTM(units=hidden_units // 2, activation='tanh',
             recurrent_activation='sigmoid', return_sequences=True,
             batch_input_shape=(batch_size, time_step, input_dim))
    )
    # model.add(Dropout(dropout_rate))

    # 二层
    model.add(
        LSTM(units=hidden_units, activation='tanh',
             recurrent_activation='sigmoid', return_sequences=True)
    )
    # model.add(Dropout(dropout_rate))

    # 三层
    model.add(
        LSTM(units=hidden_units // 2, activation='tanh',
             recurrent_activation='sigmoid', return_sequences=True)
    )
    # model.add(Dropout(dropout_rate))

    # 四层
    model.add(TimeDistributed(Dense(3)))

    # 未定义评价函数
    model.compile(loss='mean_squared_error', optimizer='nadam')

    plot_model(model, to_file='double_lstm_model.png', show_shapes=True)

    return model
