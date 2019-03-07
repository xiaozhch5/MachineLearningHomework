import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import mat
from numpy import pi
from scipy.interpolate import spline

N = 10
num_w = 10
lda = math.exp(-18)
x = [t/5 for t in range(N)]
y = [math.sin(2 * pi * i) for i in x]
plt.plot(x, y)
for j in range(N):
    y[j] = y[j] + random.gauss(0, 0.1)  # sin(x)加上高斯噪声
# 现在训练集为[x,y]
x = np.array(x)
y = np.array(y)
plt.scatter(x, y)  # 画出加上噪声之后sin函数散点图
X = np.zeros([N, num_w])
for index_1 in range(N):
    for index_2 in range(num_w):
        X[index_1, index_2] = pow(x[index_1], index_2)
XTX = mat(np.dot(X.T, X) + lda/2 * np.eye(num_w, num_w))
W = np.dot(np.dot(XTX.I, X.T), y)
print(W)
Y = np.zeros(N)
for index_3 in range(N):
    for index_4 in range(num_w):
        Y[index_3] = Y[index_3] + W[0, index_4] * pow(x[index_3], index_4)
xnew = np.linspace(x.min(), x.max(), 500)
Y_smooth = spline(x, Y, xnew)
plt.plot(xnew, Y_smooth)
plt.show()
