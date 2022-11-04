
from sklearn import linear_model
import numpy as np


# MSE(mean square error)均方误差
# theat0参数 theat1参数 xList自变量 yList因变量
def MSE(theat0 ,theat1 ,xList ,yList):
    sum = 0
    m = len(xList)
    for i in range(m):
        # xList[i] = round(xList[i],5)
        # yList[i] = round(yList[i],5)
        sum += (theat0 + theat1 * xList[i] - yList[i]) ** 2

    value = sum / ( 2 *m)
    return value


# 梯度下降算法迭代更新theat0和theat1
def gradient_descent(theat0,theat1,alpha ,xList ,yList):

    m = len(xList)
    sum0 = 0
    sum1 = 0

    for i in range(m):
        sum0 += theat0 +theat1 * xList[i] - yList[i]
        sum1 += (theat0 +theat1 * xList[i] - yList[i]) * xList[i]

    theat0 = theat0 - alpha * sum0 / m
    theat1 = theat1 - alpha * sum1 / m

    return theat0,theat1