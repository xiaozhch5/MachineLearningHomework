# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from random import choice


# 实验目的
# 找到一个点[x*, y*]使得2000个随机的点到这个点的距离最短加起来
# 梯度下降法
# 设置损失函数为E(x, y) = sum_{i = 0}^N((x_i - x)^2 + (y_i - y)^2)
# 对函数求x偏导：E'_x = sum_{i = 0}^N((x - x_i)/((x - x_i)^2 + (y - y_i)^2)^(1/2))
# 对函数求y偏导：E'_y = sum_{i = 0}^N((y - y_i)/((x - x_i)^2 + (y - y_i)^2)^(1/2))
# 随机梯度下降法
# 损失函数和偏导数相比于梯度下降法来说少了求和
N = 2000  # 在[0， 1]范围内随机生成的点数
x = [np.random.random() for i in range(N)]  # 在[0， 1]内随机生成N个点
y = [np.random.random() for j in range(N)]


# 梯度下降
def bgd(xx, yy, x_, y_):  # 输入全部样本点，返回损失函数对x和y的偏导数
    dx = 0
    dy = 0
    for index_1 in range(N):
        dx = dx + (x_ - xx[index_1]) / (((xx[index_1] - x_) ** 2 + (yy[index_1] - y_) ** 2) ** 1/2)
        dy = dy + (y_ - yy[index_1]) / (((xx[index_1] - x_) ** 2 + (yy[index_1] - y_) ** 2) ** 1/2)
    return [dx, dy]


def bgd_loss(xx, yy, x_, y_):  # 输入全部样本点，返回损失函数
    e = 0
    for index_1 in range(N):
        e = e + ((xx[index_1] - x_) ** 2 + (yy[index_1] - y_) ** 2) ** 1/2
    return e


def bgd_():
    alpha = 0.0000001  # 梯度下降法的学习率
    x_initial = 0  # 出发点[0 , 1]
    y_initial = 1
    i = 0
    x_record = np.zeros(1000000)
    y_record = np.zeros(1000000)
    while True:  # 通过梯度下降法找到点[x*, y*],当第n次和第n-1次的误差函数小于阈值时候，得到最优点
        x_record[i] = x_initial
        y_record[i] = y_initial
        e_1 = bgd_loss(x, y, x_initial, y_initial)
        [dx, dy] = bgd(x, y, x_initial, y_initial)
        # print(x_initial, y_initial)
        x_initial = x_initial - alpha * dx
        y_initial = y_initial - alpha * dy
        e_2 = bgd_loss(x, y, x_initial, y_initial)
        i += 1
        if abs(e_2 - e_1) < 0.00001:
            break
    return x_initial, y_initial, x_record, y_record, i


# 随机梯度下降
def sgd_loss(random_x, random_y, x_, y_):  # 输入随机取到的某一个样本，返回损失函数
    e = ((random_x - x_) ** 2 + (random_y - y_) ** 2) ** 1/2
    return e


def sgd(random_x, random_y, x_, y_):  # 输入随机取到的某一个样本，返回损失函数对x和y的梯度
    dx = (x_ - random_x)/(((random_x - x_) ** 2 + (random_y - y_) ** 2) ** 1/2)
    dy = (y_ - random_y)/(((random_x - x_) ** 2 + (random_y - y_) ** 2) ** 1/2)
    return [dx, dy]


def sgd_():
    alpha = 0.00001  # 学习率
    x_initial = 0  # 出发点为[0， 1]
    y_initial = 1
    i = 0
    x_record = np.zeros(1000000)
    y_record = np.zeros(1000000)
    while True:  # 通过梯度下降法找到点[x*, y*],当第n次和第n-1次的误差函数小于阈值时候，得到最优点
        x_record[i] = x_initial
        y_record[i] = y_initial
        x_choice = choice(x)
        y_choice = choice(y)
        e1 = sgd_loss(x_choice, y_choice, x_initial, y_initial)
        [dx, dy] = sgd(x_choice, y_choice, x_initial, y_initial)
        x_initial = x_initial - alpha * dx
        y_initial = y_initial - alpha * dy
        e2 = sgd_loss(x_choice, y_choice, x_initial, y_initial)
        i += 1
        if abs(e1 - e2) < 0.000001:
            break
    return x_initial, y_initial, x_record, y_record, i


if __name__ == '__main__':
    x_final_1, y_final_1, x__1, y__1, i_1 = bgd_()
    x_final_2, y_final_2, x__2, y__2, i_2 = sgd_()
    x__1 = x__1[0:i_1-1]
    y__1 = y__1[0:i_1-1]
    x__2 = x__2[0:i_2-1]
    y__2 = y__2[0:i_2-1]
    print(len(x__1), len(x__2))
    plt.figure(1)
    plt.title(' 2000 random points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)  # 画出随机生成的2000个点
    plt.figure(2)
    plt.title('BGD')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.plot(x__1, y__1, color='r', linestyle='solid')  # 画出梯度下降法的最优点寻找路径
    plt.figure(3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SGD')
    plt.scatter(x, y)
    plt.plot(x__2, y__2, color='r', linestyle='solid')  # 画出随机梯度下降法的最优点寻找路径
    plt.show()
    print(x_final_1, y_final_1)  # 打印梯度下降法找到的最优点
    print(x_final_2, y_final_2)  # 打印随机梯度下降法找到的最优点
