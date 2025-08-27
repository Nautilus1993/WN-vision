# -*- coding: utf-8 -*-
# @Author  : Zhao Yutao
# @Time    : 2024/8/22 11:40
# @Function: 构建Cislunar动力学模型及积分，状态更新
# @mails: zhaoyutao22@mails.ucas.ac.cn
import time

import matplotlib.pyplot as plt
from math import sin, cos, sqrt  # sin,cos的输入是弧度
import numpy as np
import spiceypy as spice
from scipy.integrate import odeint, solve_ivp
from utils.utils import load_furnsh
# 加载Spice内核文件
spice.tkvrsn("TOOLKIT")
load_furnsh('kernel')


#动力学模型——Cislunar 考虑地球、月球、太阳
# 引力常数（m^3/s^2）
mu_earth = 3.98600441e14  # 地球
mu_moon = 4.903e12  # 月球
mu_sun = 1.327e20  # 太阳
J2 = 0.00108263 #暂时不考虑J摄动
sun_pos, moon_pos = [], []
# 时间步长（秒）
# dt = 1.0
def pos_ems(et):
    '''
    更新获取每一et时刻下地球J2000下太阳、月球位置，地心一直为[0,0,0]
    :param et:星历时
    :return: 太阳位置、月球位置数组
    '''
    # filename = img_path.split("\\")[-1].split("_")
    # utc = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + filename[
    #     4] + ":" + filename[5] + "." + filename[6]
    # et = spice.utc2et(utc)
    # et = spice.utc2et("2024-8-15T00:00:00.000")
    # print(spice.et2utc(et,'C',3))
    sun_pos, lighttime = spice.spkpos('SUN', et, 'J2000', 'NONE', 'Earth')
    moon_pos, lighttime = spice.spkpos('MOON', et, 'J2000', 'NONE', 'Earth')
    return sun_pos*1000,moon_pos*1000

def gravitational_force(state, t):
    '''
    引力函数//未考虑非球形引力摄动
    输入当前状态，时刻，返回考虑地、月、日后的引力加速度
    :param state: 状态向量 [x, y, z, vx, vy, vz]
    :param t: 当前et时刻或者utc时刻，utc需转et
    :return: 引力加速度
    '''
    x, v = state[:3], state[3:]
    # 日月位置
    sun_pos, moon_pos = pos_ems(t)
    # 添加地日月引力
    F = np.zeros(3)
    # 添加地球引力
    r_mag_earth = np.linalg.norm(x)
    F += mu_earth * x / r_mag_earth**3
    # 添加太阳引力
    r_sun = x - sun_pos
    r_mag_sun = np.linalg.norm(r_sun)
    F += mu_sun * ( r_sun / r_mag_sun**3 + sun_pos / np.linalg.norm(sun_pos))
    # 添加月球引力
    r_moon = x - moon_pos
    r_mag_moon = np.linalg.norm(r_moon)
    F += mu_moon * ( r_moon / r_mag_moon**3 + moon_pos / np.linalg.norm(moon_pos))
    return np.array(-F)

def Cislunar(state, t):
    '''
    m/s
    '''
    x, y, z, vx, vy, vz = state[0], state[1], state[2], state[3], state[4], state[5]
    dx = np.zeros((6))
    # 速度微分项 未考虑J2摄动
    dvxdt = (-mu_earth * x / r_mag_earth**3 -
             mu_moon * (r_moon[0]/r_mag_moon**3 + moon_pos[0]/np.linalg.norm(moon_pos)**3) -
             mu_sun*(r_sun[0]/r_mag_sun**3 + sun_pos[0]/np.linalg.norm(sun_pos)**3))
    dvydt = (-mu_earth * y / r_mag_earth**3 -
             mu_moon * (r_moon[1]/r_mag_moon**3 + moon_pos[1]/np.linalg.norm(moon_pos)**3) -
             mu_sun*(r_sun[1]/r_mag_sun**3 + sun_pos[1]/np.linalg.norm(sun_pos)**3))
    dvzdt = (-mu_earth * z / r_mag_earth**3 -
             mu_moon * (r_moon[2]/r_mag_moon**3 + moon_pos[2]/np.linalg.norm(moon_pos)**3) -
             mu_sun*(r_sun[2]/r_mag_sun**3 + sun_pos[2]/np.linalg.norm(sun_pos)**3))
    dx[0] = vx
    dx[1] = vy
    dx[2] = vz
    dx[3] = dvxdt
    dx[4] = dvydt
    dx[5] = dvzdt
    return dx

def jac_F(state,delta_t,Xm,Ym,Zm,Xs,Ys,Zs):
    '''
    状态转移矩阵更新
    :param state:
    :return: 返回
    '''
    x, y, z = state[0],state[1],state[2]
    # # 探测器此刻状态距地距离
    # r_mag_earth = np.linalg.norm(state[:3])
    # # 探测器此刻到日距离
    # r_sun = state[:3] - sun_pos
    # r_mag_sun = np.linalg.norm(r_sun)
    # # 探测器此刻到月距离
    # r_moon = state[:3] - moon_pos
    # r_mag_moon = np.linalg.norm(r_moon)

    jac_ = np.zeros((6,6))
    jac_[0:3, 0:3] = np.eye(3)
    jac_[0:3, 3:6] = np.eye(3) * delta_t
    jac_[3:6, 3:6] = np.eye(3)
    F11 = mu_earth * (3 * (x**2) - r_mag_earth**2) / r_mag_earth**5 + \
          mu_moon * (3 * (r_moon[0]**2) - r_mag_moon**2) / r_mag_moon**5 +\
          mu_sun * (3 * (r_sun[0]**2) - r_mag_sun**2) / r_mag_sun**5

    F22 = mu_earth * (3 * (y**2) - r_mag_earth**2) / r_mag_earth**5 + \
          mu_moon * (3 * (r_moon[1]**2) - r_mag_moon**2) / r_mag_moon**5 + \
          mu_sun * (3 * (r_sun[1]**2) - r_mag_sun**2) / r_mag_sun**5

    F33 = mu_earth * (3 * (z**2) - r_mag_earth**2) / r_mag_earth**5 + \
          mu_moon * (3 * (r_moon[2]**2) - r_mag_moon**2) / r_mag_moon**5 + \
          mu_sun * (3 * (r_sun[2]**2) - r_mag_sun**2) / r_mag_sun**5

    F12 = mu_earth * 3 * x * y / r_mag_earth**5 + \
          mu_moon * 3 *(r_moon[0])*(r_moon[1]) / r_mag_moon**5 + \
          mu_sun * 3 *(r_sun[0])*(r_sun[1]) / r_mag_sun**5
    F21 = F12

    F13 = mu_earth * 3 * x * z / r_mag_earth**5 + \
          mu_moon * 3 *(r_moon[0])*(r_moon[2]) / r_mag_moon**5 + \
          mu_sun * 3 *(r_sun[0])*(r_sun[2]) / r_mag_sun**5
    F31 = F13

    F23 = mu_earth * 3 * y * z / r_mag_earth**5 + \
          mu_moon * 3 *(r_moon[1])*(r_moon[2]) / r_mag_moon**5 + \
          mu_sun * 3 *(r_sun[1])*(r_sun[2]) / r_mag_sun**5

    F32 = F23
    Fa = np.array([[F11,F12,F13],[F21,F22,F23],[F31,F32,F33]])
    jac_[3:6,0:3] = Fa * delta_t
    # print(jac_)
    return jac_

def Cislunar_Update6_state(X_p, delta_t, et):
    t = np.linspace(0, delta_t, 100)
    # 日月位置需要根据et实时更新
    global sun_pos,moon_pos,r_mag_earth,r_moon,r_mag_moon,r_sun,r_mag_sun
    sun_pos, moon_pos = pos_ems(et)
    # 探测器此刻状态距地距离
    r_mag_earth = np.linalg.norm(X_p[:3])
    # 探测器此刻到日距离
    r_sun = X_p[:3] - sun_pos
    r_mag_sun = np.linalg.norm(r_sun)
    # 探测器此刻到月距离
    r_moon = X_p[:3] - moon_pos
    r_mag_moon = np.linalg.norm(r_moon)
    F = jac_F(X_p, delta_t,moon_pos[0],moon_pos[1],moon_pos[2],sun_pos[0],sun_pos[1],sun_pos[2])
    solution = odeint(Cislunar, X_p, t) # 积分器时间过长
    X_prop = solution[-1, :]
    return X_prop, F, moon_pos,sun_pos
# print(Cislunar_Update6_state([-92301944.869,-293655873.151,-169433426.578,1212.453, 101.398, 27.511],33,766714536.8486029))