# -*- coding: utf-8 -*-
# @Author  : Zhao Yutao
# @Time    : 2024/12/8 17:52
# @Function: ,无迹卡尔曼滤波ukf
# @mails: zhaoyutao22@mails.ucas.ac.cn
# @Author  : Zhao Yutao
# @Time    : 2024/8/21 12:29
# @Function: 利用Cislunar动力学模型对月球观测矢量进行EKF滤波定轨
# @mails: zhaoyutao22@mails.ucas.ac.cn
# coding = utf-8
import datetime
import math
import os
import time
from math import sin, cos, sqrt  # sin,cos的输入是弧度
import spiceypy as spice
from scipy.ndimage import median_filter
from Filter.Cislunar_Update_X_prop import jac_F, Cislunar, Cislunar_Update6_state
from utils.utils import load_furnsh
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import setting.settingsPara as para

# 加载Spice内核文件
spice.tkvrsn("TOOLKIT")
load_furnsh('kernel')

class UKF_ours:
    def __init__(self,batching:bool, path = None, path2 = None):
        self.case_H_map = {
            1: "仅对月球测距",
            2: "对月球测距、转系后的赤经赤纬(方位角、俯仰角)",
            3: "对月球测距、转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角",
            4: "对月球转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角",
            5: "对地球、月球、太阳测方位角与俯仰角",
            6: "对地球、太阳测测方位角与俯仰角",
            7: "对月球、太阳测测方位角与俯仰角",
            8: "TLI观测多控制点"
        }
        self.log_file = None
        self.ukf_type = 'ED'
        # self.ukf_ = ukf()
        if batching:
            self.time_measure = []
            self.position_rho_measure = []
            self.position_a_measure = []
            self.position_b_measure = []
            self.position_rho_true = []
            self.position_a_true = []
            self.position_b_true = []
            self.position_velocity_measure = []
            self.position_x_true = []
            self.position_y_true = []
            self.position_z_true = []
            self.speed_x_true = []
            self.speed_y_true = []
            self.speed_z_true = []
            self.Pointing_Precision = []
            self.residual = []
            self.position_x_measure = []
            self.position_y_measure = []
            self.position_z_measure = []
            self.speed_x_measure = []
            self.speed_y_measure = []
            self.speed_z_measure = []
            self.position_x_prior_est = []  # X方向位置的先验估计值
            self.position_y_prior_est = []
            self.position_z_prior_est = []
            self.speed_x_prior_est = []
            self.speed_y_prior_est = []
            self.speed_z_prior_est = []
            self.position_x_posterior_est = []  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计值x位置值存入到列表中
            self.position_y_posterior_est = []
            self.position_z_posterior_est = []
            self.speed_x_posterior_est = []
            self.speed_y_posterior_est = []
            self.speed_z_posterior_est = []
            self.q_true = []
            self.X_posterior_1_list = []
            self.e_alpha_list = []
            self.e_beta_list = []
            self.s_alpha_list = []
            self.s_beta_list = []
            self.v_alpha_list = []
            self.v_beta_list = []
            self.l4_alpha_list = []
            self.l4_beta_list = []
            self.l5_alpha_list = []
            self.l5_beta_list = []
            self.dro_alpha_list = []
            self.dro_beta_list = []
            self.moon_pos = []
            self.err_3axis_list = []
            self.err_3axis_dict = {}
            if batching == True:
                self.data_batching(path,path2)
        else:
            self.time_measure = deque() # 存储当前时刻及上一时刻
            self.position_x_measure = None # 单帧估计X轴作为初值
            self.position_y_measure = None # 单帧估计Y轴作为初值
            self.position_z_measure = None # 单帧估计Z轴作为初值
            self.speed_x_measure = None  # 如何初始化速度这一选项
            self.speed_y_measure = None
            self.speed_z_measure = None
            self.position_x_prior_est = None # X方向位置的先验估计值
            self.position_y_prior_est = None # Y方向位置的先验估计值
            self.position_z_prior_est = None # Z方向位置的先验估计值
            self.speed_x_prior_est = None # X方向速度的先验估计值
            self.speed_y_prior_est = None # Y方向速度的先验估计值
            self.speed_z_prior_est = None # Z方向速度的先验估计值
            self.position_x_posterior_est = None # 最优估计值x位置值
            self.position_y_posterior_est = None # 最优估计值y位置值
            self.position_z_posterior_est = None # 最优估计值z位置值
            self.speed_x_posterior_est = None
            self.speed_y_posterior_est = None
            self.speed_z_posterior_est = None
            self.X_posterior_1 = None # 测量量减去最优估计值--先验残差
            self.P_posterior = None
            self.X_posterior = None
            self.es_q = None # 四元数估计
            self.e_core_x = None # 地球质心提取像素点
            self.e_core_y = None
            self.m_core_x = None # 月球质心提取像素点
            self.m_core_y = None
            self.solve_outlier_moon = deque()
            self.solve_outlier_earth = deque()
            self.notsolve_outlier_moon = deque()
            self.notsolve_outlier_earth = deque()
            self.time_usemeasure_earth = deque()
            self.time_usemeasure_moon = deque()
            '''
            分时拉齐观测量
            '''
            self.mode = para.ekf_mode
            self.e_rho = None
            self.e_alpha = None
            self.e_beta = None
            self.m_rho = None
            self.m_alpha = None
            self.m_beta = None
            self.time_front = None # 标记上一不同观测量的时间戳
            self.flag_front = '' # 标记上一不同观测量的类型是地球还是月球或者金星、木星，现在只对地月
            self.flag_front_front = ''
            self.flag_time_update = None # 标记是否可以进行时间更新
            '''
            地月空间多控制点测角信息
            '''
            self.l4_alpha = None
            self.l4_beta = None
            self.l5_alpha = None
            self.l5_beta = None
            self.dro_alpha = None
            self.dro_beta = None
            self.venus_alpha = None
            self.venus_beta = None
            self.jupiter_alpha = None
            self.jupiter_beta = None
            '''
            等待过程中各测量量是否更新
            '''
            self.m_update_flag = False
            self.e_update_flag = False
            self.l4_update_flag = False
            self.l5_update_flag = False
            self.dro_update_flag = False
            self.venus_update_flag = False
            self.moon_pos = []
            self.P_position_err = []
            self.P_vec_err = []
            self.P_obv = []
            if self.mode == 1:
                self.flag_time_update = True
            else:
                self.flag_time_update = False
    def set_ukf_type(self,type):
        self.ukf_type = type
        self.save_dir = r"C:\Users\zhaoy\PycharmProjects\EarthMoon\TestData\UKFresult\\" + para.testdata.split("/")[-1].split('\\')[-1] + '_'+type+'_1'
        for i in range(1,10000):
            if not os.path.exists(self.save_dir.rsplit('_',1)[0] + '_' + str(i)):
                self.save_dir = self.save_dir.rsplit('_',1)[0] + '_' + str(i)
                os.mkdir(self.save_dir)
                break
    def init_list(self):
        self.position_x_prior_est = []  # X方向位置的先验估计值
        self.position_y_prior_est = []
        self.position_z_prior_est = []
        self.speed_x_prior_est = []
        self.speed_y_prior_est = []
        self.speed_z_prior_est = []
        self.position_x_posterior_est = []  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计值x位置值存入到列表中
        self.position_y_posterior_est = []
        self.position_z_posterior_est = []
        self.speed_x_posterior_est = []
        self.speed_y_posterior_est = []
        self.speed_z_posterior_est = []
        self.X_posterior_1_list = []
        self.err_3axis_list = []
    def data_batching(self, path, path2):
        '''
        数据批处理，提取测量文件中所有数据变量为滤波做准备
        :param path: 测量文件地址
        :return: 19类变量
        '''
        # q_file_name = os.listdir(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\data\EllipseRCNNDatasets\Moonnormal")
        # q_file_name = sorted(q_file_name,key=natural_sort_key)
        file = open(path)
        gtmoon_file = open(path2,'r')
        for line in gtmoon_file.readlines():
            curline =np.array(list(map(float, line.strip().split(" ")[1:4])))
            curline = -curline
            self.position_rho_true.append(np.linalg.norm(curline)*1000)
            self.position_a_true.append(math.atan2(curline[1],curline[0]))
            self.position_b_true.append(math.atan2(curline[2],math.sqrt(curline[1]**2+curline[0]**2)))

        for line in file.readlines():
            curLine = line.strip().split(" ")
            # 取出相机观测数据
            self.time_measure.append(float(curLine[0]))
            self.position_rho_measure.append(float(curLine[1])*1000)
            self.position_a_measure.append(float(curLine[2]))
            self.position_b_measure.append(float(curLine[3]))
            self.position_velocity_measure.append(float(curLine[4])*1000)
            self.position_x_true.append(float(curLine[5])*1000)
            self.position_y_true.append(float(curLine[6])*1000)
            self.position_z_true.append(float(curLine[7])*1000)
            self.speed_x_true.append(float(curLine[8])*1000)
            self.speed_y_true.append(float(curLine[9])*1000)
            self.speed_z_true.append(float(curLine[10])*1000)
            # self.Pointing_Precision.append(float(curLine[11]))
            self.residual.append(float(curLine[12]))
            self.position_x_measure.append(float(curLine[13])*1000)
            self.position_y_measure.append(float(curLine[14])*1000)
            self.position_z_measure.append(float(curLine[15])*1000)
            self.speed_x_measure.append(float(curLine[16])*1000)
            self.speed_y_measure.append(float(curLine[17])*1000)
            self.speed_z_measure.append(float(curLine[18])*1000)
            self.e_alpha_list.append(float(curLine[19]))
            self.e_beta_list.append(float(curLine[20]))
            self.s_alpha_list.append(float(curLine[21]))
            self.s_beta_list.append(float(curLine[22]))
            self.v_alpha_list.append(float(curLine[23]))
            self.v_beta_list.append(float(curLine[24]))
            self.l4_alpha_list.append(float(curLine[25]))
            self.l4_beta_list.append(float(curLine[26]))
            self.l5_alpha_list.append(float(curLine[27]))
            self.l5_beta_list.append(float(curLine[28]))
            self.dro_alpha_list.append(float(curLine[29]))
            self.dro_beta_list.append(float(curLine[30]))

    def H_moondis(self,Px,Py,Pz,moon_pos,position_rho_):
        return  np.array([[(Px-moon_pos[0])/position_rho_,(Py-moon_pos[1])/position_rho_,(Pz-moon_pos[2])/position_rho_,0,0,0]])

    def symb_Jac_Fun(self):
        '''
        对H矩阵求偏导
        '''
        import sympy as sp
        from sympy import symbols, Function, diff
        x, y, z, X, Y, Z = symbols('Px Py Pz Xm Ym Zm')
        f0 = Function('f')(x, y, z, X, Y, Z)
        f0 = sp.sqrt((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2)
        df0_dx = diff(f0, x)
        df0_dy = diff(f0, y)
        df0_dz = diff(f0, z)
        f1 = Function('f')(x, y, X, Y)
        f1 = sp.atan2(Y - y, X - x)
        df1_dx = diff(f1, x)
        df1_dy = diff(f1, y)
        df1_dz = diff(f1, z)
        # -----------------------***********-----------------
        f2 = Function('f')(x, y, z, X, Y, Z)
        f2 = sp.atan2(Z - z, sp.sqrt((X - x) ** 2 + (Y - y) ** 2))
        df2_dx = diff(f2, x)
        df2_dy = diff(f2, y)
        df2_dz = diff(f2, z)
        # -----------------------***********-----------------
        H = np.array([[df0_dx,df0_dy,df0_dz,0,0,0],
                      [df1_dx,df1_dy,df1_dz,0,0,0],
                      [df2_dx,df2_dy,df2_dz,0,0,0]])
        print(df2_dx)
        print(df2_dy)
        print(df2_dz)
        return H

    def init_single_ukf(self):
        self.time_measure = deque()  # 存储当前时刻及上一时刻
        self.position_x_measure = None  # 单帧估计X轴作为初值
        self.position_y_measure = None  # 单帧估计Y轴作为初值
        self.position_z_measure = None  # 单帧估计Z轴作为初值
        self.speed_x_measure = None  # 如何初始化速度这一选项
        self.speed_y_measure = None
        self.speed_z_measure = None
        self.position_x_prior_est = None  # X方向位置的先验估计值
        self.position_y_prior_est = None  # Y方向位置的先验估计值
        self.position_z_prior_est = None  # Z方向位置的先验估计值
        self.speed_x_prior_est = None  # X方向速度的先验估计值
        self.speed_y_prior_est = None  # Y方向速度的先验估计值
        self.speed_z_prior_est = None  # Z方向速度的先验估计值
        self.position_x_posterior_est = None  # 最优估计值x位置值
        self.position_y_posterior_est = None  # 最优估计值y位置值
        self.position_z_posterior_est = None  # 最优估计值z位置值
        self.speed_x_posterior_est = None
        self.speed_y_posterior_est = None
        self.speed_z_posterior_est = None
        self.X_posterior_1 = None  # 测量量减去最优估计值--先验残差
        self.P_posterior = None
        self.X_posterior = None
        self.es_q = None  # 四元数估计
        self.core_x = None  # 地球月球质心提取像素点
        self.core_y = None
        self.e_rho = None
        self.e_alpha = None
        self.e_beta = None
        self.m_rho = None
        self.m_alpha = None
        self.m_beta = None
        self.P_position_err = []
        self.P_vec_err = []
        self.P_obv = []
        # self.m_core_x = None  # 月球质心提取像素点
        # self.m_core_y = None


    def ukf_update_single(self, timestamps, singlex = None, singley = None, singlez = None,
                          position_x_measure = None, position_y_measure = None, position_z_measure = None,
                          e_rho = None, e_alpha = None, e_beta = None, m_rho = None, m_alpha = None, m_beta = None,
                          rho_err=100000, m2c_err=0.0001, P_xyz_cov=1000000, P_Vxyz_cov=0.0000001,
                          Q_xyz_cov=1e-8, Q_Vxyz_cov=1e-8,kappa=0, alpha=0.1,beta=2):
        '''
        接收每帧的测量量并进入EKF,并不是每段时间下都可以观测到地月，如果观测只有1个，那么就按照1个的来
        '''
        # alpha=0.1: Sigma点分布的缩放参数，控制分布的广度
        # beta=2.0: 用于捕获先验分布的尾部重量，beta=2是最佳选择对于高斯分布
        # kappa=0: 二次项的缩放参数，影响Sigma点分布
        self.time_measure.append(timestamps)
        P = np.eye(6)
        P[0:3, 0:3] = np.eye(3) * P_xyz_cov ** 2
        P[3:6, 3:6] = np.eye(3) * P_Vxyz_cov ** 2
        if len(self.time_measure) == 1:
            self.position_x_measure = position_x_measure
            self.position_y_measure = position_y_measure
            self.position_z_measure = position_z_measure
            pv, lighttime = spice.spkezr(para.sat_id_str,timestamps, 'J2000', 'None', 'Earth')
            self.speed_x_measure = pv[3]*1000  # 如何初始化速度这一选项
            self.speed_y_measure = pv[4]*1000  # 暂时利用真值进行初始速度
            self.speed_z_measure = pv[5]*1000
            self.X_posterior = np.array([self.position_x_measure,
                                    self.position_y_measure,
                                    self.position_z_measure,
                                    self.speed_x_measure,
                                    self.speed_y_measure,
                                    self.speed_z_measure])  # X_posterior表示上一时刻的最优估计值
            self.P_posterior = np.array(P)  # P_posterior是继续更新最优解的协方差矩阵
            if type(self.es_q) is np.ndarray:
                logname = para.testdata.split("/")[-1].split('\\')[-1] + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_" +self.ukf_type + "_stares.txt"
                self.log_file = open(os.path.join(self.save_dir,logname), 'a')
            else:
                logname = para.testdata.split("/")[-1].split('\\')[-1] + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_" +self.ukf_type +  "_stargt.txt"
                self.log_file = open(os.path.join(self.save_dir,logname), 'a')
        else:
            windows_epoch = 1000
            if self.e_rho is None and e_rho is not None:
                '''
                剔除或者中值拉低孤立值对滤波稳定性的影响
                基于窗口的方式，第一帧不加入，也是为了降低第一帧如果错误带来的影响，降低对第一帧的置信度
                窗口的大小设置为100存储量，同一观测量，当大于100/2的时候即可对孤立值进行检测
                '''
                if len(self.solve_outlier_earth) >= windows_epoch:
                    self.solve_outlier_earth.popleft() # 超过100推出左边值
                    self.notsolve_outlier_earth.popleft()
                    self.time_usemeasure_earth.popleft()
                self.time_usemeasure_earth.append(timestamps)
                self.notsolve_outlier_earth.append([e_rho, e_alpha, e_beta])  # 未考虑数据点离群，做对比实验

                '''
                对窗口内值进行孤立值检测,如果孤立，则置为None，或者中值滤波
                Z-score标准化后离群点检测对于测量量较小波动的稳定性滤波友好，
                中值滤波对于测量量非常波动的较为优化，中值窗口值可设大点
                结论：观测量的数据预处理是有效的，数据清洗
                '''
                if len(self.solve_outlier_earth) >= 25:
                    # 计算Z-Score
                    mean = np.mean(np.array(self.notsolve_outlier_earth)[:, 1])
                    std_dev = np.std(np.array(self.notsolve_outlier_earth)[:, 1])
                    z_socre = (np.array(self.notsolve_outlier_earth)[:, 1] - mean) / std_dev
                    # 识别离群点
                    # print(np.abs(z_socre))
                    if np.abs(z_socre[-1]) > 2:
                        self.e_rho = None
                        self.e_alpha = None
                        self.e_beta = None
                    else:
                        self.e_rho = e_rho
                        self.e_alpha = e_alpha
                        self.e_beta = e_beta

                else:
                    self.solve_outlier_earth.append([e_rho, e_alpha, e_beta])  # 加入新值
                    self.e_rho = e_rho
                    self.e_alpha = e_alpha
                    self.e_beta = e_beta
                self.e_rho = e_rho
                self.e_alpha = e_alpha
                self.e_beta = e_beta

            if self.m_rho is None and m_rho is not None:

                if len(self.solve_outlier_moon) >= windows_epoch:
                    self.solve_outlier_moon.popleft() # 超过100推出左边值
                    self.notsolve_outlier_moon.popleft()
                    self.time_usemeasure_moon.popleft()
                self.notsolve_outlier_moon.append([m_rho, m_alpha, m_beta])  # 未考虑数据点离群，做对比实验
                self.time_usemeasure_moon.append(timestamps)
                if len(self.solve_outlier_moon) >= 25:
                    mean = np.mean(np.array(self.notsolve_outlier_moon)[:, 1])
                    std_dev = np.std(np.array(self.notsolve_outlier_moon)[:, 1])
                    z_socre = (np.array(self.notsolve_outlier_moon)[:, 1] - mean) / std_dev
                    if np.abs(z_socre[-1]) > 2:
                        self.m_rho = None
                        self.m_alpha = None
                        self.m_beta = None
                    else:
                        self.m_rho = m_rho
                        self.m_alpha = m_alpha
                        self.m_beta = m_beta

                else:
                    self.solve_outlier_moon.append([m_rho, m_alpha, m_beta])  # 加入新值
                # print('+++++++++++++++++')
                    self.m_rho = m_rho
                    self.m_alpha = m_alpha
                    self.m_beta = m_beta
                self.m_rho = m_rho
                self.m_alpha = m_alpha
                self.m_beta = m_beta
            delta_t = self.time_measure[1] - self.time_measure[0]
            self.time_measure.popleft()

            # # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
            Q = np.eye(6)
            Q[0:3, 0:3] = np.eye(3) * Q_xyz_cov ** 2
            Q[3:6, 3:6] = np.eye(3) * Q_Vxyz_cov ** 2
            R = np.eye(4) * m2c_err**2
            R[0, 0] = 0.001 ** 2
            R[1, 1] = 0.001 ** 2
            # 计算状态估计协方差矩阵P
            if self.e_alpha is not None and self.m_alpha is not None:
                Z_measure = np.array([[self.m_alpha, self.m_beta, self.e_alpha, self.e_beta]]).T
                # # 这个时候需要拉齐上一帧标记的观测量到这一帧
                if self.flag_front == 'M' and self.time_front is not None and self.X_posterior is not None:
                    # 上一观测量是月球则更新对月球的观测量
                    X_prior_front, F_front, moon_pos_front, sun_pos_front = Cislunar_Update6_state(self.X_posterior,
                                                                                                   timestamps - self.time_front,
                                                                                                   self.time_front)
                    T_m2c = moon_pos_front - X_prior_front[0:3]
                    self.m_rho = np.linalg.norm(T_m2c)
                    self.m_alpha = math.atan2(T_m2c[1], T_m2c[0])
                    self.m_beta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
                elif self.flag_front == 'E' and self.time_front is not None and self.X_posterior is not None:
                    X_prior_front, F_front, moon_pos_front, sun_pos_front = Cislunar_Update6_state(self.X_posterior,
                                                                                                   timestamps - self.time_front,
                                                                                                   self.time_front)

                    T_m2c = - X_prior_front[0:3]
                    self.e_rho = np.linalg.norm(T_m2c)
                    self.e_alpha = math.atan2(T_m2c[1], T_m2c[0])
                    self.e_beta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
                self.flag_time_update = True  # 当地月观测量都有的时候，可以进行时间更新，然后把上一帧标记的观测量置空，更新上一帧观测量为现在观测量
                if self.flag_front == 'M':
                    self.flag_front_front = 'M'
                    self.flag_front = 'E'
                    self.time_front = timestamps  # 这个时间戳会更新成最新进入的观测量的时间戳，然后成为下一个的上一观测量
                    self.m_rho = None
                    self.m_alpha = None
                    self.m_beta = None
                    # print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMEEEEEEEEEEEEEEEEEEEEEEEEEEE')
                else:
                    self.flag_front = 'M'
                    self.flag_front_front = 'E'
                    self.time_front = timestamps
                    self.e_rho = None
                    self.e_alpha = None
                    self.e_beta = None
                    # print('EEEEEEEEEEEEEEEEEEEEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
            elif self.e_alpha is None:
                self.time_front = timestamps
                self.flag_front = 'M'  # 如果持续是同一观测量，则更新同一观测量，时间仍然为标记上一帧为异测量量的时间戳
                self.flag_front_front = 'M'
                self.flag_time_update = False
                # print('***********************************')
            elif self.m_alpha is None:
                self.time_front = timestamps
                self.flag_front = 'E'
                self.flag_front_front = 'E'
                self.flag_time_update = False
                # print("/////////////////////////////////////////")

            if self.flag_time_update == True:

                def generate_sigma_points(x, P, alpha, kappa, beta):
                    '''
                    经过UT变换球的Sigma点生成函数及其权值
                    '''
                    n = len(x)
                    # print(n)
                    # print(x)
                    lambda_ = alpha ** 2 * (n + kappa) - n
                    sigma_points = np.zeros((2 * n + 1, n))
                    # print(sigma_points)
                    # print(x)
                    sigma_points[0] = x
                    # P = np.around(P,decimals=0)
                    U = np.linalg.cholesky((n + lambda_) * P)
                    for i in range(n):
                        sigma_points[i + 1] = x + U[:, i]
                        sigma_points[n + i + 1] = x - U[:, i]

                    sigma_points = np.zeros((2 * n + 1, n))
                    sigma_points[0] = x
                    for k in range(n):
                        sigma_points[k + 1] = np.subtract(x, -U[k])
                        sigma_points[n + k + 1] = np.subtract(x, U[k])
                    Wm = np.zeros(13)
                    Wc = np.zeros(13)
                    for i in range(13):
                        Wm[i] = 1 / (2 * (n + lambda_))
                        Wc[i] = 1 / (2 * (n + lambda_))
                    Wm[0] = lambda_ / (6 + lambda_)
                    Wc[0] = Wm[0] + (1 - alpha ** 2 + beta)
                    return sigma_points, Wm, Wc

                def propagate_sigma_points(sigma_points, delta_t, last_timestamps):
                    '''
                    Sigma点传播函数，计算2n+1个点集的一步预测
                    '''
                    psp = []
                    for point in sigma_points:
                        X_prior, F, moon_pos, sun_pos = Cislunar_Update6_state(point, delta_t, last_timestamps)
                        self.moon_pos.append(moon_pos)
                        psp.append(X_prior)
                    return np.array(psp)

                def predict_state_and_covariance(sigma_points, Wm, Wc):
                    '''
                    预测状态和协方差函数，相当于EKF的先验估计
                    '''
                    # print(sigma_points[0])
                    X_pred = np.zeros(6)
                    P_pred = np.zeros((6, 6))
                    for i in range(13):
                        X_pred = X_pred + Wm[i] * sigma_points[i]
                    # print(Wc)
                    for i in range(13):
                        P_pred = P_pred + Wc[i] * (sigma_points[i] - X_pred).T @ (sigma_points[i] - X_pred) + Q
                    return X_pred, P_pred

                def ut_2(x, P, alpha, kappa, beta):
                    '''
                    第二次UT变换
                    '''
                    n = len(x)
                    lambda_ = alpha ** 2 * (n + kappa) - n
                    sigma_points = np.zeros((2 * n + 1, n))
                    sigma_points[0] = x
                    U, s, V = np.linalg.svd(P)
                    for i in range(n):
                        sigma_points[i + 1] = x + sqrt(6 + lambda_) * U[:, i] * sqrt(s[i])
                        sigma_points[n + i + 1] = x - sqrt(6 + lambda_) * U[:, i] * sqrt(s[i])
                    return sigma_points

                def H(X_prior):
                    X_m2c = self.moon_pos[-1] - X_prior[0:3]
                    m_alpha = math.atan2(X_m2c[1], X_m2c[0])
                    m_beta = math.atan2(X_m2c[2], math.sqrt(X_m2c[0] ** 2 + X_m2c[1] ** 2))
                    e_alpha = math.atan2(-X_prior[1], -X_prior[0])
                    e_beta = math.atan2(-X_prior[2], np.sqrt(X_prior[0] ** 2 + X_prior[1] ** 2))
                    Z_X = np.array([[m_alpha, m_beta, e_alpha, e_beta]]).T
                    return Z_X

                def update_observation(sigma_points_ut2, h, Wm, Wc):
                    '''
                    观测更新函数s
                    '''
                    Z_pred = np.zeros((4, 1))
                    P_zkzk = np.zeros((4, 4))
                    P_xkzk = np.zeros((6, 4))
                    yk = []
                    for i in range(13):
                        Z_pred = Z_pred + Wm[i] * h(sigma_points_ut2[i])
                    # print('P_zkzk',P_zkzk)
                    for i in range(13):
                        P_xkzk = P_xkzk + Wc[i] * np.array([sigma_points_ut2[i] - X_pred]).T @ (
                                    h(sigma_points_ut2[i]) - Z_pred).T
                        yk.append(h(sigma_points_ut2[i]) - Z_pred)
                    vi = np.median(yk, axis=0)
                    vi_index = 7
                    for i in range(13):
                        if (vi == yk[i]).all():
                            vi_index = i
                    qvi = abs(1 / Wc[vi_index])
                    # print(qvi)
                    dk = 1.4826 * np.linalg.norm(vi) / sqrt(qvi)
                    svk = np.linalg.norm(vi) / (dk * sqrt(qvi))
                    rk = None
                    k0 = 2
                    k1 = 5
                    if svk <= k0:
                        rk = 1
                    elif svk <= k1:
                        rk = k0 / svk * ((k1 - svk) / (k1 - k0))
                    else:
                        rk = 1e-30
                    for i in range(13):
                        P_zkzk_kangcha = P_zkzk + Wc[i] * (h(sigma_points_ut2[i]) - Z_pred) @ (
                            (h(sigma_points_ut2[i]) - Z_pred).T) + R / rk
                        P_zkzk = P_zkzk + Wc[i] * (h(sigma_points_ut2[i]) - Z_pred) @ (
                            (h(sigma_points_ut2[i]) - Z_pred).T) + R
                    if np.linalg.cond(P_zkzk) < 1e15:
                        K_kangcha = P_xkzk @ np.linalg.inv(P_zkzk_kangcha)
                    else:
                        K_kangcha = P_xkzk @ np.linalg.inv(P_zkzk_kangcha) * rk
                    K_kangcha = P_xkzk @ np.linalg.inv(P_zkzk)
                    return Z_pred, K_kangcha, P_zkzk_kangcha

                def update_state(K, X_pred, P_pred, Z_measure, z_pred, P_zkzk):
                    '''
                    状态更新函数
                    '''
                    # print(X_pred)
                    # print(K @ (Z_measure - z_pred))
                    # print(K, X_pred, P_pred, Z_measure, z_pred, P_zkzk)
                    self.X_posterior_1 = (Z_measure - z_pred).T.squeeze()
                    x_update = X_pred + (K @ (Z_measure - z_pred)).T
                    P_update = P_pred - np.dot(K, np.dot(P_zkzk, K.T))
                    return x_update.squeeze(), P_update

                    # ------------------- 下面开始进行预测和更新，来回不断的迭代 -------------------------
                    # 求前后两帧的时间差，数据包中的时间戳单位为微秒，处以1e6，转换为秒
                    # ---------------------- 时间更新  -------------------------
                    # 先验估计值与状态转移矩阵
                sigma_points, Wm, Wc = generate_sigma_points(self.X_posterior, self.P_posterior, alpha, kappa, beta)
                sigma_points_prop = propagate_sigma_points(sigma_points, delta_t, self.time_measure[-1])
                X_pred, P_pred = predict_state_and_covariance(sigma_points_prop, Wm, Wc)
                # sigma_points_ut2 = ut_2(X_pred, P_pred, alpha, kappa, beta)
                sigma_points_ut2 = sigma_points
                z_pred, K, P_zkzk = update_observation(sigma_points_ut2, H, Wm, Wc)
                self.X_posterior,P_posterior = update_state(K, X_pred, P_pred, Z_measure, z_pred, P_zkzk)
                # self.P_posterior = P_posterior
                # print('X_posterior', self.X_posterior)
                self.P_obv.append([sqrt(P[0, 0] / P_posterior[0, 0]),
                                   sqrt(P[1, 1] / P_posterior[1, 1]),
                                   sqrt(P[2, 2] / P_posterior[2, 2]),
                                   sqrt(P[3, 3] / P_posterior[3, 3]),
                                   sqrt(P[4, 4] / P_posterior[4, 4]),
                                   sqrt(P[5, 5] / P_posterior[5, 5])])
                self.P_position_err.append(
                    np.sqrt(np.array([P_posterior[0, 0] + P_posterior[1, 1] + P_posterior[2, 2]])))
                self.P_vec_err.append(np.sqrt(np.array([P_posterior[3, 3] + P_posterior[4, 4] + P_posterior[5, 5]])))
                # 最优估计值从m换成km
                self.position_x_posterior_est = self.X_posterior[0] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
                self.position_y_posterior_est = self.X_posterior[1] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
                self.position_z_posterior_est = self.X_posterior[2] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z位置值存入到列表中
                self.speed_x_posterior_est = self.X_posterior[3] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
                self.speed_y_posterior_est = self.X_posterior[4] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中
                self.speed_z_posterior_est = self.X_posterior[5]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z速度值存入到列表中

                X_posterior_1_str = ' '.join(str(i) for i in self.X_posterior_1)
                str_es_q = None
                if type(self.es_q) is np.ndarray:
                    str_es_q = ' '.join(str(i) for i in self.es_q)
                    str_es_q += ' '
                else:
                    str_es_q = ''
                # if self.core_x is None:
                self.log_file.write(str(self.time_measure[0]) + " " +
                                        str(self.position_x_posterior_est) + " " +
                                        str(self.position_y_posterior_est) + " " +
                                        str(self.position_z_posterior_est) + " " +
                                        str(self.speed_x_posterior_est) + " " +
                                        str(self.speed_y_posterior_est) + " " +
                                        str(self.speed_z_posterior_est) + " " +
                                        str(singlex) + " " +
                                        str(singley) + " " +
                                        str(singlez) + " " +
                                        str(self.core_x) + " " +
                                        str(self.core_y) + " " +
                                        str_es_q +
                                        X_posterior_1_str + "\n")


    def ukf_update_batch(self,
                           rho_err=100000,
                           m2c_err=1e-4,
                           P_yxz_cov=50000,
                           P_Vxyz_cov=1e-8,
                           Q_xyz_cov=1e-16,
                           Q_Vxyz_cov=1e-16, kappa=0, alpha=100,beta=2):
        # --------------------------- 初始化 -------------------------
        # 用第2帧测量数据初始化
        # alpha=0.1: Sigma点分布的缩放参数，控制分布的广度
        # beta=2.0: 用于捕获先验分布的尾部重量，beta=2是最佳选择对于高斯分布
        # kappa=0: 二次项的缩放参数，影响Sigma点分布
        # 初始化X和P
        X0 = np.array([self.position_x_measure[1],
                       self.position_y_measure[1],
                       self.position_z_measure[1],
                       self.speed_x_measure[1],
                       self.speed_y_measure[1],
                       self.speed_z_measure[1]])
        self.position_x_prior_est.append(X0[0])
        self.position_y_prior_est.append(X0[1])
        self.position_z_prior_est.append(X0[2])
        self.speed_x_prior_est.append(X0[3])
        self.speed_y_prior_est.append(X0[4])
        self.speed_z_prior_est.append(X0[5])
        # 用第1帧初始时间戳
        last_timestamp_ = self.time_measure[1]
        # 状态估计协方差矩阵P初始化（其实就是初始化最优解的协方差矩阵）
        P = np.eye(6)
        P[0:3, 0:3] = np.eye(3) * P_yxz_cov ** 2
        P[3:6, 3:6] = np.eye(3) * P_Vxyz_cov ** 2
        X_posterior = X0 # X_posterior表示上一时刻的最优估计值
        P_posterior = P  # P_posterior是继续更新最优解的协方差矩阵
        # 将初始化后的数据依次送入(即从第三帧速度往里送)
        delta_t = 0
        len_ = len(self.time_measure)
        # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
        Q = np.eye(6)
        Q[0:3, 0:3] = np.eye(3) * Q_xyz_cov ** 2
        Q[3:6, 3:6] = np.eye(3) * Q_Vxyz_cov ** 2
        R = np.eye(4) * m2c_err ** 2
        R = np.eye(4) * m2c_err ** 2
        R[0, 0] = 0.001 ** 2
        R[1, 1] = 0.001 ** 2


        def generate_sigma_points(x, P, alpha, kappa, beta):
            '''
            经过UT变换球的Sigma点生成函数及其权值
            '''
            n = len(x)
            # print(n)
            # print(x)
            lambda_ = alpha ** 2 * (n + kappa) - n
            sigma_points = np.zeros((2 * n + 1, n))
            # print(sigma_points)
            # print(x)
            sigma_points[0] = x
            U, s, V = np.linalg.svd(P)
            for i in range(n):
                sigma_points[i + 1] = x + sqrt(6 + lambda_) * U[:, i] * sqrt(s[i])
                sigma_points[n + i + 1] = x - sqrt(6 + lambda_) * U[:, i] * sqrt(s[i])

            sigma_points = np.zeros((2 * n + 1, n))
            sigma_points[0] = x
            for k in range(n):
                # pylint: disable=bad-whitespace
                sigma_points[k + 1] = np.subtract(x, -U[k])
                sigma_points[n + k + 1] = np.subtract(x, U[k])
            Wm = np.zeros(13)
            Wc = np.zeros(13)
            for i in range(13):
                Wm[i] = 1/(2*(n+lambda_))
                Wc[i] = 1/(2*(n+lambda_))
            Wm[0] = lambda_/(6+lambda_)
            Wc[0] = Wm[0] + (1-alpha**2 + beta)
            # print(Wm)
            # print(Wc)
            # assert np.sum(wtab) == 1
            return sigma_points, Wm, Wc

        def propagate_sigma_points(sigma_points, delta_t, last_timestamps):
            '''
            Sigma点传播函数，计算2n+1个点集的一步预测
            '''
            psp = []
            for point in sigma_points:
                X_prior, F, moon_pos, sun_pos = Cislunar_Update6_state(point, delta_t, last_timestamps)
                self.moon_pos.append(moon_pos)
                psp.append(X_prior)
            return np.array(psp)

        def predict_state_and_covariance(sigma_points, Wm,Wc):
            '''
            预测状态和协方差函数，相当于EKF的先验估计
            '''
            # print(sigma_points[0])
            X_pred = np.zeros(6)
            P_pred = np.zeros((6,6))
            for i in range(13):
                X_pred = X_pred + Wm[i] * sigma_points[i]
            for i in range(13):
                P_pred = P_pred + Wc[i] * (sigma_points[i] - X_pred).T @ (sigma_points[i] - X_pred) + Q
            # print('P_pred',P_pred)
            return X_pred, P_pred

        def ut_2(x, P, alpha, kappa, beta):
            '''
            第二次UT变换
            '''
            n = len(x)
            lambda_ = alpha ** 2 * (n + kappa) - n
            sigma_points = np.zeros((2 * n + 1, n))
            sigma_points[0] = x
            # U = np.linalg.cholesky((n + lambda_) * P)
            U,s,V = np.linalg.svd(P)
            # print(s)
            # print('U',U)
            for i in range(n):
                sigma_points[i + 1] = x + sqrt(6+lambda_) * U[:, i] * sqrt(s[i])
                sigma_points[n + i + 1] = x - sqrt(6+lambda_) * U[:, i] * sqrt(s[i])
            return sigma_points

        def H(X_prior):
            X_m2c = self.moon_pos[-1] - X_prior[0:3]
            m_alpha = math.atan2(X_m2c[1], X_m2c[0])
            m_beta = math.atan2(X_m2c[2], math.sqrt(X_m2c[0] ** 2 + X_m2c[1] ** 2))
            e_alpha = math.atan2(-X_prior[1], -X_prior[0])
            e_beta = math.atan2(-X_prior[2], np.sqrt(X_prior[0] ** 2 + X_prior[1] ** 2))
            Z_X = np.array([[m_alpha, m_beta, e_alpha, e_beta]]).T
            return Z_X

        def update_observation(sigma_points_ut2, h, Wm,Wc):
            '''
            观测更新函数s
            '''
            Z_pred = np.zeros((4,1))
            P_zkzk = np.zeros((4,4))
            P_xkzk = np.zeros((6,4))
            yk = []
            for i in range(13):
                Z_pred = Z_pred + Wm[i] * h(sigma_points_ut2[i])
            for i in range(13):
                P_xkzk = P_xkzk + Wc[i] * np.array([sigma_points_ut2[i] - X_pred]).T @ (h(sigma_points_ut2[i]) - Z_pred).T
                yk.append(h(sigma_points_ut2[i]) - Z_pred)
            vi = np.median(yk,axis=0)
            vi_index = 7
            for i in range(13):
                if (vi == yk[i]).all():
                    vi_index = i
            qvi = abs(1/Wc[vi_index])
            dk = 1.4826 * np.linalg.norm(vi)/sqrt(qvi)
            svk = np.linalg.norm(vi)/(dk*sqrt(qvi))
            rk = None
            k0 = 2
            k1 = 5
            if svk <= k0:
                rk = 1
            elif svk <= k1:
                rk =  k0/svk * ((k1-svk)/(k1-k0))
            else:
                rk = 1e-30
            for i in range(13):
                P_zkzk_kangcha = P_zkzk + Wc[i] * (h(sigma_points_ut2[i]) - Z_pred) @ ((h(sigma_points_ut2[i]) - Z_pred).T) + R / rk
                P_zkzk = P_zkzk + Wc[i] * (h(sigma_points_ut2[i]) - Z_pred) @ (
                    (h(sigma_points_ut2[i]) - Z_pred).T) + R
            if np.linalg.cond(P_zkzk) < 1e15:
                K_kangcha = P_xkzk @ np.linalg.inv(P_zkzk_kangcha)
            else:
                K_kangcha = P_xkzk @ np.linalg.inv(P_zkzk) * rk
            # K = P_xkzk @ np.linalg.inv(P_zkzk)
            return Z_pred, K_kangcha, P_zkzk_kangcha

        def update_state(K, X_pred, P_pred,Z_measure, z_pred, P_zkzk):
            '''
            状态更新函数
            '''
            # print(X_pred)
            # print(K @ (Z_measure - z_pred))
            x_update = X_pred + (K @ (Z_measure - z_pred)).T
            P_update = P_pred - np.dot(K, np.dot(P_zkzk, K.T))
            return x_update.squeeze(), P_update
        '''
        𝑘0、𝑘1 为阈值参数, 通常 𝑘0 取1.5 ∼ 2.0, 𝑘1 取
        3.0 ∼ 8.5, 文中取 𝑘0 = 2, 𝑘1 = 4; 𝑠𝑣𝑘、𝜎𝑘 分别为标准
        化残差和基于中位数计算的方差因子
        '''

        for i in range(2, len_):
            # ------------------- 下面开始进行预测和更新，来回不断的迭代 -------------------------
            # 求前后两帧的时间差，数据包中的时间戳单位为微秒，处以1e6，转换为秒
            # ---------------------- 时间更新  -------------------------
            # 先验估计值与状态转移矩阵
            delta_t = self.time_measure[i] - last_timestamp_
            Z_measure = np.array([[self.position_a_measure[i],
                                 self.position_b_measure[i],
                                 self.e_alpha_list[i],
                                 self.e_beta_list[i]]]).T
            # print('P_posterior_one',P_posterior)
            sigma_points, Wm,Wc = generate_sigma_points(X_posterior, P_posterior, alpha, kappa, beta)
            sigma_points_prop = propagate_sigma_points(sigma_points, delta_t, last_timestamp_)
            X_pred, P_pred = predict_state_and_covariance(sigma_points_prop, Wm,Wc)
            # print(X_pred)
            # sigma_points_ut2 = ut_2(X_pred, P_pred, alpha, kappa, beta)
            sigma_points_ut2 = sigma_points
            z_pred, K, P_zkzk = update_observation(sigma_points_ut2, H, Wm,Wc)
            X_posterior, P_posterior = update_state(K, X_pred, P_pred,Z_measure, z_pred,P_zkzk)
            # print('X_posterior',X_posterior)
            # print('P_posterior',P_posterior)
            last_timestamp_ = self.time_measure[i]

            # 最优估计值从m换成km
            self.position_x_posterior_est.append(X_posterior[0])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
            self.position_y_posterior_est.append(X_posterior[1])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
            self.position_z_posterior_est.append(X_posterior[2])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z位置值存入到列表中
            self.speed_x_posterior_est.append(X_posterior[3])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
            self.speed_y_posterior_est.append(X_posterior[4])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中
            self.speed_z_posterior_est.append(X_posterior[5])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z速度值存入到列表中

            # cos_a = self.position_x_true[i]*1000 * X_posterior[0] + self.position_y_true[i]*1000 * X_posterior[1] + self.position_x_true[i]*1000 * X_posterior[2]
            # cos_b = np.linalg.norm([self.position_x_true[i]*1000,self.position_y_true[i]*1000,self.position_z_true[i]*1000]) * np.linalg.norm(X_posterior[0:3])
            # cos = cos_a / cos_b
            # if cos > 1:
            #     cos = 1
            # elif cos < -1:
            #     cos = -1
            cos = 1
            self.Pointing_Precision.append(abs(math.acos(cos) * 180 / math.pi))
        # 可视化显示
        if True:
            ore_precision = median_filter(self.Pointing_Precision, 25)
            print(ore_precision)
            print("反推质心提取误差/pixel:" + str(
                300 * math.tan(np.deg2rad(np.sqrt(np.mean(np.array(ore_precision[25:]) ** 2)))) /(5.236 / 8 * 1024)))
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 坐标图像中显示中文
            plt.rcParams['axes.unicode_minus'] = False
            # 绘制真值与最优估计值图
            # self.xyz_true_posterior_est()
            # 绘制单帧定位的位速误差图
            self.single_err_plot(rho_err,
                                 m2c_err,
                                 P_yxz_cov,
                                 P_Vxyz_cov,
                                 Q_xyz_cov,
                                 Q_Vxyz_cov,
                                 delta_t,
                                 self.time_measure[0],
                                 self.time_measure[len_ - 1])
            # # 绘制角分辨率与测量量误差图
            # self.Pointing_Precision_residual_plot()
            # 绘制测量量残差
            # self.X_posterior_1_plot(case_H)
            # 绘制3d的xyz图
            # self.threed_xyz()

    def threed_xyz(self):
        # 绘制x-y-z图
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        ax.set_title('3d_image_show')  # 设置本图名称
        self.moon_pos = np.array(self.moon_pos)
        for i in range(len(self.position_x_true)-2):
            if i == 0:
                plt.pause(3)
            if i % 200 == 0 or i < 30:
                ax.cla()
                ax.plot(self.position_x_true[2:i+2], self.position_y_true[2:i+2], self.position_z_true[2:i+2],label="gt", c='b')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                # ax.scatter(self.position_x_measure[:i], self.position_y_measure[:i], self.position_z_measure[:i],label="single", c='g')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.plot(self.position_x_posterior_est[:i], self.position_y_posterior_est[:i], self.position_z_posterior_est[:i],label="ukf", c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.plot(self.moon_pos[:i,0], self.moon_pos[:i,1], self.moon_pos[:i,2],label="moon", c='c')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.scatter(0, 0, 0, label="earth", c='b', s=100, marker="o")
                ax.scatter(self.position_x_posterior_est[i], self.position_y_posterior_est[i], self.position_z_posterior_est[i], label="ukf_update",
                           c='y', s=100, marker="^")  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.set_xlabel('X')  # 设置x坐标轴
                ax.set_ylabel('Y')  # 设置y坐标轴
                ax.set_zlabel('Z')  # 设置z坐标轴
                ax.legend()
                plt.pause(0.001)

            if i == len(self.position_x_true) - 3:
                plt.show()
    def xyz_true_posterior_est(self):
        # 绘制真值与最优估计值图
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs[0, 0].plot(self.position_x_true[2:], "-", label="位置x_实际值", linewidth=1)
        axs[0, 0].plot(self.position_x_posterior_est, "-", label="位置x_最优估计值", linewidth=1)

        axs[0, 0].set_title("位置x")
        axs[0, 0].set_xlabel('k')
        axs[0, 0].legend()

        axs[0, 1].plot(self.position_y_true[2:], "-", label="位置y_实际值", linewidth=1)
        axs[0, 1].plot(self.position_y_posterior_est, "-", label="位置y_最优估计值", linewidth=1)
        # axs[0, 1].plot(position_y_posterior_est, "-", label="位置y_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        axs[0, 1].set_title("位置y")
        axs[0, 1].set_xlabel('k')
        axs[0, 1].legend()

        axs[0, 2].plot(self.position_z_true[2:], "-", label="位置z_实际值", linewidth=1)
        axs[0, 2].plot(self.position_z_posterior_est, "-", label="位置z_最优估计值", linewidth=1)
        # axs[0, 2].plot(position_z_posterior_est, "-", label="位置z_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        axs[0, 2].set_title("位置z")
        axs[0, 2].set_xlabel('k')
        axs[0, 2].legend()

        axs[1, 0].plot(self.speed_x_true[2:], "-", label="速度x_实际值", linewidth=1)
        axs[1, 0].plot(self.speed_x_posterior_est, "-", label="速度x_最优估计值", linewidth=1)
        # axs[1, 0].plot(speed_x_posterior_est[20:], "-", label="速度x_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        axs[1, 0].set_title("速度x")
        axs[1, 0].set_xlabel('k')
        axs[1, 0].legend()

        axs[1, 1].plot(self.speed_y_true[2:], "-", label="速度y_实际值", linewidth=1)
        axs[1, 1].plot(self.speed_y_posterior_est, "-", label="速度y_最优估计值", linewidth=1)
        # axs[1, 1].plot(speed_y_posterior_est[20:], "-", label="速度y_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        axs[1, 1].set_title("速度y")
        axs[1, 1].set_xlabel('k')
        axs[1, 1].legend()

        axs[1, 2].plot(self.speed_z_true[2:], "-", label="速度z_实际值", linewidth=1)
        axs[1, 2].plot(self.speed_z_posterior_est, "-", label="速度z_最优估计值", linewidth=1)
        # axs[1, 2].plot(speed_z_posterior_est[20:], "-", label="速度z_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        axs[1, 2].set_title("速度z")
        axs[1, 2].set_xlabel('k')
        axs[1, 2].legend()
        plt.show()
    def single_err_plot(self,rho_err,
                   m2c_err,
                   P_yxz_cov ,
                   P_Vxyz_cov,
                   Q_xyz_cov,
                   Q_Vxyz_cov,
                   epoch,
                   start_time,
                   end_time,keypoints=None):
        '''
        # 单帧定位误差画图
        '''
        if keypoints is None:
            case = self.case_H_map[4]
        else:
            case = keypoints
        err_px = []
        err_py = []
        err_pz = []
        err_vx = []
        err_vy = []
        err_vz = []
        v_err_threshold = 0.5
        v_err_threshold_id = 0
        flag = 0
        if len(open("ExRecord.csv",encoding='utf-8').readlines()) < 1:
            csv_a = open("ExRecord.csv",'a',encoding='utf-8')
            csv_a.write(
                "实验时间"+ "," +
                "测距测量量误差"+ "," +
                "像素精度" + "," +
                "初始位置误差" + "," +
                "初始速度误差" + "," +
                "位置过程噪声" + "," +
                "速度过程噪声" + "," +
                "积分步长" + "," +
                "开始历元" + "," +
                "结束历元" + "," +
                "收敛步" + "," +
                "观测量case" + "," +
                "X轴RSME" + "," +
                "Y轴RSME" + "," +
                "Z轴RSME" + "," +
                "Vx轴RSME" + "," +
                "Vy轴RSME" + "," +
                "Vz轴RSME" + "\n"
            )
            csv_a.close()
        csv_save = open("ExRecord.csv",'a',encoding='utf-8')
        # self.speed_x_posterior_est = median_filter(self.speed_z_posterior_est, 25)
        # self.speed_y_posterior_est = median_filter(self.speed_z_posterior_est, 25)
        # self.speed_z_posterior_est = median_filter(self.speed_z_posterior_est, 25)
        # print(len(self.position_x_posterior_est))
        # print(len(self.position_x_true))
        '''
        做一个中值滑动窗口就行，对观测矩阵做滑窗复杂且效果不一定好，数据平滑给远距离，波动较大数据友好
        '''
        for i in range(len(self.position_x_posterior_est)):
            err_px.append(abs(self.position_x_posterior_est[i]-self.position_x_true[i+2]))
            err_py.append(abs(self.position_y_posterior_est[i]-self.position_y_true[i+2]))
            err_pz.append(abs(self.position_z_posterior_est[i]-self.position_z_true[i+2]))
            err_vx.append(abs(self.speed_x_posterior_est[i]-self.speed_x_true[i+2]))
            err_vy.append(abs(self.speed_y_posterior_est[i]-self.speed_y_true[i+2]))
            err_vz.append(abs(self.speed_z_posterior_est[i]-self.speed_z_true[i+2]))
            if np.linalg.norm([err_vx[i],err_vy[i],err_vz[i]]) <= v_err_threshold and flag == 0:
                v_err_threshold_id = i
                flag = 1
            self.err_3axis_list.append(np.linalg.norm(np.array([err_px[i],err_py[i],err_pz[i]])))
        self.err_3axis_dict[keypoints] = np.array(self.err_3axis_list)
        # plt.plot(np.log10(self.err_3axis_list))
        # plt.title(keypoints)
        # plt.show()
        v_err_threshold_id = 400
        err_px = median_filter(err_px,25)
        err_py = median_filter(err_py,20)
        err_pz = median_filter(err_pz,25)
        def rmse(predictions, targets):
            pt = median_filter(predictions[v_err_threshold_id:] - targets[v_err_threshold_id:],25)
            return np.sqrt((pt ** 2).mean())
        print("从第",v_err_threshold_id, "步收敛稳定————")
        print(self.case_H_map[4], "的RMSE:")
        print("X:", rmse(np.array(self.position_x_posterior_est),np.array(self.position_x_true[2:])))
        print("Y:", rmse(np.array(self.position_y_posterior_est),np.array(self.position_y_true[2:])))
        print("Z:", rmse(np.array(self.position_z_posterior_est),np.array(self.position_z_true[2:])))
        print("Vx:", rmse(np.array(self.speed_x_posterior_est),np.array(self.speed_x_true[2:])))
        print("Vy:", rmse(np.array(self.speed_y_posterior_est),np.array(self.speed_y_true[2:])))
        print("Vz:", rmse(np.array(self.speed_z_posterior_est),np.array(self.speed_z_true[2:])))
        print("XYZ:",np.linalg.norm(np.array([rmse(np.array(self.position_x_posterior_est),np.array(self.position_x_true[2:])),
                                              rmse(np.array(self.position_y_posterior_est),np.array(self.position_y_true[2:])),
                                              rmse(np.array(self.position_z_posterior_est),np.array(self.position_z_true[2:]))])))
        csv_save.write(
            str(datetime.datetime.now())+","+
            str(rho_err)+","+
            str(m2c_err)+","+
            str(P_yxz_cov)+","+
            str(P_Vxyz_cov)+","+
            str(Q_xyz_cov)+","+
            str(Q_Vxyz_cov)+","+
            str(epoch)+","+
            str(start_time)+","+
            str(end_time)+","+
            str(v_err_threshold_id)+","+
            case+","+
            str(rmse(np.array(self.position_x_posterior_est),np.array(self.position_x_true[2:])))+","+
            str(rmse(np.array(self.position_y_posterior_est),np.array(self.position_y_true[2:]))) + "," +
            str(rmse(np.array(self.position_z_posterior_est),np.array(self.position_z_true[2:]))) + "," +
            str(np.sqrt(np.mean((err_vx[v_err_threshold_id:] - np.mean(err_vx[v_err_threshold_id:]))**2))) + "," +
            str(np.sqrt(np.mean((err_vy[v_err_threshold_id:] - np.mean(err_vy[v_err_threshold_id:]))**2))) + "," +
            str(np.sqrt(np.mean((err_vz[v_err_threshold_id:] - np.mean(err_vz[v_err_threshold_id:]))**2))) + "\n"
        )
        csv_save.close()

        return v_err_threshold_id
    def X_posterior_1_plot(self,case_H):
        '''
        先验残差图
        '''
        len_ = len(self.X_posterior_1_list[0])
        fig, axs = plt.subplots(4, 4, figsize=(20, 15))
        for i in range(len_):
            a = math.floor(i/4)
            b = i % 4
            label_title = "测量量 "+str(i)+" 的残差"
            axs[a, b].plot(np.array(self.X_posterior_1_list)[:,i], "-", label=label_title, linewidth=1)
            axs[a, b].set_title(self.case_H_map[case_H])
            axs[a, b].set_xlabel('k')
            axs[a, b].legend()
        plt.show()

    def Pointing_Precision_residual_plot(self):
        '''
        角分辨率与测量量残差图
        '''
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].plot(self.Pointing_Precision, "-", label="角分辨率", linewidth=1)
        axs[0].set_title("角分辨率°")
        axs[0].set_xlabel('k')
        axs[0].legend()

        axs[1].plot(self.residual, "-", label="测量量残差", linewidth=1)
        axs[1].set_title("测量量残差km")
        axs[1].set_xlabel('k')
        axs[1].legend()
        plt.show()

    def err_3axis_plot(self):
        # import seaborn as sns
        # sns.barplot(x='time', y='导航精度',  width=0.5,
        #             palette=['cornflowerblue', 'aqua', 'deepskyblue'])
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.keys())
        i =  0
        for k,v in self.err_3axis_dict.items():
            i += 1
            # print(k)
            plt.plot(self.time_measure[2:],v,label=k,color=mcolors.TABLEAU_COLORS[colors[i]])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    fpath = "simulation_data/simulation_data_2w.txt"
    fpath2 = 'simulation_data/sisimulation_data_2w_gt_moon.txt'
    ukf = UKF_ours(True, fpath, fpath2)
    ukf.ukf_update_batch()
    # ukf.init_list()
