from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
import numpy as np
from keras.utils import plot_model

batch_size = 5  # 训练样本批次
time_steps = 30  # 时间步长
data_dim = 3  # x的特征数目
hidden_units = 256  # 隐藏层神经元个数
dropout_rate = 0.2  # 正则化层概率

# 期望输入数据尺寸: (batch_size, time_steps, data_dim)
# 请注意，我们必须提供完整的 batch_input_shape, 因为网络是有状态的。
# 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
# 关于 k 批的思考，多段路径对应批中的路径，不会发生混乱
model = Sequential()
# 个人猜测 recurrent_activation为门激活函数, activation为输入 c~的激活函数
model.add(LSTM(units=hidden_units, activation='tanh',
               recurrent_activation='relu', return_sequences=True,
               stateful=True, batch_input_shape=(batch_size, time_steps, data_dim)))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=hidden_units, activation='tanh',
               recurrent_activation='relu', return_sequences=True,
               stateful=True))
model.add(Dropout(dropout_rate))
model.add(TimeDistributed(Dense(3)))

# 未定义评价函数
model.compile(loss='mean_squared_error', optimizer='nadam')

plot_model(model, to_file='stateful_double_lstm_model.png', show_shapes=True)