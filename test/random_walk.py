#!/user/bin/env python
# coding=utf-8
"""
@project : double_layer_lstm
@author  : Apollo
@file   : random_walk.py
@ide    : PyCharm
@time   : 2021-01-23 15:38:40
@desc   ： 随机游走测试-1
"""
import matplotlib.pyplot as plt
import numpy as np

# rect=[0.1,5.0,0.1,0.1]
fig = plt.figure()

# time span
T = 10
# drift factor飘移率
mu = 0.1
# volatility波动率
sigma = 0.04
# t=0初试价
S0 = 50
# length of steps
dt = 0.5
N = round(T / dt)
t = np.linspace(0, T, N)

# 布朗运动
W = np.random.standard_normal(size=N)
W = np.cumsum(W) * np.sqrt(dt)

X = (mu - 0.5 * sigma ** 2) * t + sigma * W

S = S0 * np.exp(X)

plt.plot(t, S, lw=2)
plt.show()
