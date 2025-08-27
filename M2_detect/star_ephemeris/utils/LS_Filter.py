#coding=utf-8
from scipy import signal
import math
import numpy as np
from scipy import linalg
from scipy.optimize import leastsq
#做优化，最小二乘或者其他滤波，将中间跳点平稳输出，需要实时引用
from scipy.ndimage import gaussian_filter1d
#滑动平均滤波法 （又称：递推平均滤波法），它把连续取N个采样值看成一个队列 ，队列的长度固定为N ，每次采样到一个新数据放入队尾，并扔掉原来队首的一次数据(先进先出原则) 。把队列中的N个数据进行算术平均运算，就可获得新的滤波结果。
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


#高斯平滑
def Gaussian_euler(euler_Matrix):
    X = gaussian_filter1d(euler_Matrix[:,0],1)
    Y = gaussian_filter1d(euler_Matrix[:,1],1)
    Z = gaussian_filter1d(euler_Matrix[:,2],1)
    return X[len(euler_Matrix)-1],Y[len(euler_Matrix)-1],Z[len(euler_Matrix)-1]
#最小二乘平滑
def LS_euler(euler_Matrix):
    X = euler_Matrix[:,0]
    Y = euler_Matrix[:,1]
    Z = euler_Matrix[:,2]

    # return 返回一个三元组，新RPY

def func(x, p):
    A, k, theta = p
    return A * np.sin(2 * np.pi * k * x + theta)


def residuals(p, y, x):
    return y - func(x, p)

if __name__ == '__main__':
    pass
    # x = np.linspace(0, -2 * np.pi, 100)
    # A, k, theta = 10, 0.34, math.pi/6                     # 真实数据的函数参数
    # y0 = func(x, [A, k, theta])                           # 真实数据
    # y1 = y0 + 2 * np.random.randn(len(x))                 # 加入噪声之后的实验数据
    # p0 = [7, 0.2, 0]                                      # 第一次猜测的函数拟合参数
    #
    # plsq = leastsq(residuals, p0, args=(y1, x))
    # print("真实参数:", [A, k, theta])
    # print("拟合参数:", plsq[0])
    #
    # import matplotlib.pyplot as plt
    # import pylab as pl
    #
    # plt.rcParams['font.sans-serif'] = ['SimHei']          # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
    # pl.plot(x, y0, marker='+', label=u"真实数据")
    # pl.plot(x, y1, marker='^', label=u"带噪声的实验数据")
    # pl.plot(x, func(x, plsq[0]), label=u"拟合数据")
    # pl.legend()
    # pl.show()
