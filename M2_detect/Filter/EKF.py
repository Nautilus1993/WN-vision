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
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import setting.settingsPara as para
from utils.utils import load_furnsh
# 加载Spice内核文件
spice.tkvrsn("TOOLKIT")
load_furnsh('kernel')
class EKF_ours:
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
        self.ekf_type = 'ED'
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
            self.P_position_err = []
            self.P_vec_err = []
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
            self.P_position_err = []
            self.P_vec_err = []
            self.P_obv = []
            if self.mode == 1:
                self.flag_time_update = True
            else:
                self.flag_time_update = False
    def set_ekf_type(self,type):
        self.ekf_type = type
        self.save_dir = "TestData/EKFresult/" + para.testdata.split("/")[-1].split('\\')[-1] + '_'+type+'_1'
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
        self.P_position_err = []
        self.P_vec_err = []
        self.moon_pos = []
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

        # 数据变量初始化
        # 读取测量数据
        # for i in range(560):
        #     self.q_true.append(list(map(float, q_file_name[i].split("_")[7:11])))

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

    def init_single_ekf(self):
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


    def ekf_update_single(self, timestamps, singlex = None, singley = None, singlez = None,
                          position_x_measure = None, position_y_measure = None, position_z_measure = None,
                          e_rho = None, e_alpha = None, e_beta = None, m_rho = None, m_alpha = None, m_beta = None,
                          rho_err=100000, m2c_err=0.0001, P_xyz_cov=1000000, P_Vxyz_cov=1e-8, Q_xyz_cov=1e-8, Q_Vxyz_cov=0.000001):
        '''
        接收每帧的测量量并进入EKF,并不是每段时间下都可以观测到地月，如果观测只有1个，那么就按照1个的来
        '''
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
                logname = para.testdata.split("/")[-1].split('\\')[-1] + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_" +self.ekf_type + "_stares.txt"
                self.log_file = open(os.path.join(self.save_dir,logname), 'a')
            else:
                logname = para.testdata.split("/")[-1].split('\\')[-1] + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_" +self.ekf_type +  "_stargt.txt"
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
                    # e_rho_mid = median_filter(np.array(self.notsolve_outlier_earth)[:,0], 25)
                    # e_alpha_mid = median_filter(np.array(self.notsolve_outlier_earth)[:, 1], 25)
                    # e_beta_mid = median_filter(np.array(self.notsolve_outlier_earth)[:, 2], 25)
                    # self.solve_outlier_earth.append([e_rho_mid[-1], e_alpha_mid[-1], e_beta_mid[-1]])  # 加入新值
                    # self.e_rho = e_rho_mid[-1]
                    # self.e_alpha = e_alpha_mid[-1]
                    # self.e_beta = e_beta_mid[-1]
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

                # 以上无效操作
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
                    # m_rho_mid = median_filter(np.array(self.notsolve_outlier_moon)[:, 0], 25)
                    # m_alpha_mid = median_filter(np.array(self.notsolve_outlier_moon)[:, 1], 25)
                    # m_beta_mid = median_filter(np.array(self.notsolve_outlier_moon)[:, 2], 25)
                    # self.solve_outlier_moon.append([m_rho_mid[-1], m_alpha_mid[-1], m_beta_mid[-1]])  # 加入新值
                    # self.m_rho = m_rho_mid[-1]
                    # self.m_alpha = m_alpha_mid[-1]
                    # self.m_beta = m_beta_mid[-1]
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
                # 以上无效操作
                self.m_rho = m_rho
                self.m_alpha = m_alpha
                self.m_beta = m_beta
            delta_t = self.time_measure[1] - self.time_measure[0]
            self.time_measure.popleft()
            # 状态估计协方差矩阵P初始化（其实就是初始化最优解的协方差矩阵）
            # P = np.eye(6)
            # P[0:3, 0:3] = np.eye(3) * P_xyz_cov ** 2
            # P[3:6, 3:6] = np.eye(3) * P_Vxyz_cov ** 2
            # self.P_posterior = np.array(P)  # P_posterior是继续更新最优解的协方差矩阵
            X_prior, F, moon_pos, sun_pos = Cislunar_Update6_state(self.X_posterior, delta_t, self.time_measure[0])
            X_m2c = moon_pos - X_prior[0:3]
            m_rho_rk = np.linalg.norm(X_m2c)
            m_alpha_rk = math.atan2(X_m2c[1], X_m2c[0])
            m_beta_rk = math.atan2(X_m2c[2], math.sqrt(X_m2c[0] ** 2 + X_m2c[1] ** 2))
            e_rho_rk = np.linalg.norm(X_prior[0:3])
            e_alpha_rk = math.atan2(-X_prior[1], -X_prior[0])
            e_beta_rk = math.atan2(-X_prior[2], np.sqrt(X_prior[0] ** 2 + X_prior[1] ** 2))
            # # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
            Q = np.eye(6)
            Q[0:3, 0:3] = np.eye(3) * Q_xyz_cov ** 2
            Q[3:6, 3:6] = np.eye(3) * Q_Vxyz_cov ** 2
            # 计算状态估计协方差矩阵P
            P_prior_1 = np.dot(F, self.P_posterior)  # P_posterior是上一时刻最优估计的协方差矩阵    # P_prior_1就为公式中的（F.Pk-1）
            P_prior = np.dot(P_prior_1, F.T) + Q  # P_prior是得出当前的先验估计协方差矩阵      # Q是过程协方差
            # ------------------- R|K|H|Z更新  ------------------------
            [Px, Py, Pz] = X_prior[0:3]
            [Xm, Ym, Zm] = moon_pos
            # 避免被除数为0
            position_rho_m = sqrt((Xm - Px) ** 2 + (Ym - Py) ** 2 + (Zm - Pz) ** 2)
            position_rho_e = sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
            if position_rho_m < 1e-8:
                position_rho_m = 1e-8
            if position_rho_e < 1e-8:
                position_rho_e = 1e-8
            # 线性化(将非线性转为线性) 测量的协方差矩阵R，一般厂家给提供，R中的值越小，说明测量的越准确。
            Z_X, Z_measure = None, None
            if self.mode == 1:
                if self.e_alpha == None:
                    # case1： 对月观测 方位角+俯仰角+距离
                    R = np.eye(3) * m2c_err ** 2
                    R[0, 0] = rho_err ** 2
                    H = np.array([[(Px - Xm) / position_rho_m, (Py - Ym) / position_rho_m, (Pz - Zm) / position_rho_m, 0, 0, 0],
                         [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),-(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                         [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),0, 0, 0]])
                    Z_X = np.array([m_rho_rk, m_alpha_rk, m_beta_rk])
                    Z_measure = np.array([self.m_rho, self.m_alpha, self.m_beta])
                    # R = np.eye(2) * m2c_err ** 2
                    # H = np.array(
                    #     [[-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                    #       -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                    #      [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                    #                  (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                    #       (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                    #                   (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                    #       -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                    #       0, 0, 0]])
                    # Z_X = np.array([m_alpha_rk, m_beta_rk])
                    # Z_measure = np.array([m_alpha, m_beta])
                elif self.m_alpha == None:
                    # case2: 对地观测 方位角+俯仰角+距离
                    R = np.eye(3) * m2c_err ** 2
                    R[0, 0] = rho_err ** 2
                    H = np.array([[Px / position_rho_e, Py / position_rho_e, Pz / position_rho_e, 0, 0, 0],
                         [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                         [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                    Z_X = np.array([e_rho_rk, e_alpha_rk, e_beta_rk])
                    Z_measure = np.array([self.e_rho, self.e_alpha, self.e_beta])
                    # R = np.eye(2) * m2c_err ** 2
                    # H = np.array([[-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                    #               [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                    #                Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                    #                -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                    # Z_X = np.array([e_alpha_rk, e_beta_rk])
                    # Z_measure = np.array([e_alpha, e_beta])
                else:
                    # case3: 对地月观测
                    R = np.eye(6) * m2c_err ** 2
                    R[0, 0] = rho_err ** 2
                    R[3, 3] = rho_err ** 2
                    H = np.array([[(Px - Xm) / position_rho_m, (Py - Ym) / position_rho_m, (Pz - Zm) / position_rho_m, 0, 0, 0],
                                [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                                  [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                                   (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                                   -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2), 0, 0, 0],
                                  [Px / position_rho_e, Py / position_rho_e, Pz / position_rho_e, 0, 0, 0],
                                  [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                                  [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                                   Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                                   -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                    Z_X = np.array([m_rho_rk, m_alpha_rk, m_beta_rk,e_rho_rk , e_alpha_rk, e_beta_rk])
                    Z_measure = np.array([self.m_rho, self.m_alpha, self.m_beta, self.e_rho, self.e_alpha, self.e_beta])
                    # 计算卡尔曼增益
            else:
                if self.e_alpha is not None and self.m_alpha is not None:
                    # # 这个时候需要拉齐上一帧标记的观测量到这一帧
                    if self.flag_front == 'M' and self.time_front is not None and self.X_posterior is not None:
                        # 上一观测量是月球则更新对月球的观测量
                        X_prior_front, F_front, moon_pos_front, sun_pos_front = Cislunar_Update6_state(self.X_posterior,
                                                                                                       timestamps - self.time_front,
                                                                                                       self.time_front)
                        T_m2c = moon_pos_front -X_prior_front[0:3]
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

                    R = np.eye(4) * m2c_err ** 2
                    R[0, 0] = 0.001 ** 2
                    R[1, 1] = 0.001 ** 2
                    # [(Px - Xm) / position_rho_m, (Py - Ym) / position_rho_m, (Pz - Zm) / position_rho_m, 0, 0, 0],
                    H = np.array(
                        [
                         [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                          -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                         [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                     (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2), 0, 0, 0],
                         [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                         [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                    Z_X = np.array([m_alpha_rk, m_beta_rk, e_alpha_rk, e_beta_rk])
                    Z_measure = np.array([self.m_alpha, self.m_beta, self.e_alpha, self.e_beta])
                    self.flag_time_update = True # 当地月观测量都有的时候，可以进行时间更新，然后把上一帧标记的观测量置空，更新上一帧观测量为现在观测量
                    if self.flag_front == 'M':
                        self.flag_front_front = 'M'
                        self.flag_front = 'E'
                        self.time_front = timestamps # 这个时间戳会更新成最新进入的观测量的时间戳，然后成为下一个的上一观测量
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
                    self.flag_front = 'M' # 如果持续是同一观测量，则更新同一观测量，时间仍然为标记上一帧为异测量量的时间戳
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
                k1 = np.dot(P_prior, H.T)  # P_prior是得出当前的先验估计协方差矩阵
                k2 = np.dot(np.dot(H, P_prior), H.T) + R  # R是测量的协方差矩阵
                K = np.dot(k1, np.linalg.inv(k2))  # np.linalg.inv()：矩阵求逆   # K就是当前时刻的卡尔曼增益
                # 测量值
                '''
                X_posterior_1 = Z_measure - np.dot(H, X_prior) X_prior表示根据上一时刻的最优估计值得到当前的估计值
                由于观测矩阵的非线性程度太高，因此不能利用H矩阵计算观测 # 用当前X_prior的位置更新残差后项，重新计算对应的观测值
                '''
                self.X_posterior_1 = Z_measure - Z_X
                # if abs(self.X_posterior_1[-1]) < 0.0025:
                    # self.X_posterior_1 = Z_measure - np.dot(H, X_prior)
                self.X_posterior = X_prior + np.dot(K, self.X_posterior_1)  # X_posterior是根据估计值及当前时刻的观测值融合到一体得到的最优估计值
                # else:
                #     self.X_posterior = X_prior
                # 最优估计值从m换成km
                self.position_x_posterior_est = self.X_posterior[0] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
                self.position_y_posterior_est = self.X_posterior[1] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
                self.position_z_posterior_est = self.X_posterior[2]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z位置值存入到列表中
                self.speed_x_posterior_est = self.X_posterior[3]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
                self.speed_y_posterior_est = self.X_posterior[4]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中
                self.speed_z_posterior_est = self.X_posterior[5]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z速度值存入到列表中
                # print(np.linalg.norm([self.position_x_posterior_est[i-2] - self.position_x_true[i-1],
                #                       self.position_y_posterior_est[i-2] - self.position_y_true[i-1],
                #                       self.position_z_posterior_est[i-2] - self.position_z_true[i-1]]))
                '''
                更新状态估计协方差矩阵P     （其实就是继续更新最优解的协方差矩阵）
                # 经过测试，每次协方差矩阵不进行更新，一直维持原噪声就可以，H矩阵最大的作用是用于计算卡尔曼增益
                '''
                P_posterior_1 = np.eye(6) - np.dot(K, H)  # np.eye(4)返回一个4维数组，对角线上为1，其他地方为0，其实就是一个单位矩阵
                P_posterior = np.dot(P_posterior_1, P_prior)  # P_posterior是继续更新最优解的协方差矩阵  # P_prior是得出的当前的先验估计协方差矩阵
                # self.P_posterior = P_prior - np.dot(np.dot(K, H),P_prior)
                self.P_obv.append([sqrt(P[0,0]/P_posterior[0,0]),
                                  sqrt(P[1,1]/P_posterior[1,1]),
                                  sqrt(P[2,2]/P_posterior[2,2]),
                                  sqrt(P[3,3]/P_posterior[3,3]),
                                  sqrt(P[4,4]/P_posterior[4,4]),
                                  sqrt(P[5,5]/P_posterior[5,5])])
                self.P_position_err.append(
                    np.sqrt(np.array([P_posterior[0, 0] + P_posterior[1, 1] + P_posterior[2, 2]])))
                self.P_vec_err.append(np.sqrt(np.array([P_posterior[3, 3] + P_posterior[4, 4] + P_posterior[5, 5]])))

                str_es_q = None
                X_posterior_1_str = ' '.join(str(i) for i in self.X_posterior_1)
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
                # else:
                #     self.log_file.write(str(self.time_measure[0]) + " " +
                #                         str(self.position_x_posterior_est) + " " +
                #                         str(self.position_y_posterior_est) + " " +
                #                         str(self.position_z_posterior_est) + " " +
                #                         str(self.speed_x_posterior_est) + " " +
                #                         str(self.speed_y_posterior_est) + " " +
                #                         str(self.speed_z_posterior_est) + " " +
                #                         str(singlex) + " " +
                #                         str(singley) + " " +
                #                         str(singlez) + " " +
                #                         str(self.core_x) + " " +
                #                         str(self.core_y) + " " +
                #                         str_es_q +
                #                         X_posterior_1_str + "\n")
    def ekf_update_single_multi(self, timestamps,single_x = None, single_y = None, single_z = None,
                                position_x_measure = None, position_y_measure = None, position_z_measure = None,
                                m_rho=None, m_alpha=None, m_beta=None,
                                e_alpha = None, e_beta = None,
                                l4_alpha = None, l4_beta = None,
                                l5_alpha = None, l5_beta = None,
                                dro_alpha = None, dro_beta = None,
                                venus_alpha = None, venus_beta = None,
                                jupiter_alpha = None, jupiter_beta = None,
                                rho_err=100000, m2c_err=1e10, P_xyz_cov=50000,
                                P_Vxyz_cov=1e-16, Q_xyz_cov=1e-16, Q_Vxyz_cov=1e-16, keypoints='em45dv',select_flag='T'):
        '''
        接收每帧的测量量并进入EKF,并不是每段时间下都可以观测到地月，如果观测只有1个，那么就按照1个的来
        '''
        self.time_measure.append(timestamps)
        if len(self.time_measure) == 1:
            self.position_x_measure = position_x_measure
            self.position_y_measure = position_y_measure
            self.position_z_measure = position_z_measure
            if select_flag=='T':
                pv, lighttime = spice.spkezr('10002', timestamps, 'J2000', 'None', 'Earth')
            else:
                pv, lighttime = spice.spkezr('10003',timestamps, 'J2000', 'None', 'Earth')
            self.speed_x_measure = pv[3]*1000  # 如何初始化速度这一选项
            self.speed_y_measure = pv[4]*1000  # 暂时利用真值进行初始速度
            self.speed_z_measure = pv[5]*1000
            self.X_posterior = np.array([self.position_x_measure,
                                        self.position_y_measure,
                                        self.position_z_measure,
                                        self.speed_x_measure,
                                        self.speed_y_measure,
                                        self.speed_z_measure])  # X_posterior表示上一时刻的最优估计值
            logname = str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_mult.txt"
            self.log_file = open(r"../TestData/EKFresult/" + logname, 'a')
        else:
            if self.m_rho is None and m_rho is not None and 'm' in keypoints:
                print('1111111111111')
                self.m_rho = m_rho
                self.m_alpha = m_alpha
                self.m_beta = m_beta
                self.m_updata_flag = True
            if self.e_alpha is None and e_alpha is not None and 'e' in keypoints:
                print('2222222222222')
                self.e_alpha = e_alpha
                self.e_beta = e_beta
                self.e_update_flag = True
            if self.l4_alpha is None and l4_alpha is not None and '4' in keypoints:
                print('3333333333333')
                self.l4_alpha = l4_alpha
                self.l4_beta = l4_beta
                self.l4_update_flag = True
            if self.l5_alpha is None and l5_alpha is not None and '5' in keypoints:
                print('4444444444444')
                self.l5_alpha = l5_alpha
                self.l5_beta = l5_beta
                self.l5_update_flag = True
            if self.dro_alpha is None and dro_alpha is not None and 'd' in keypoints:
                print('5555555555555')
                self.dro_alpha = dro_alpha
                self.dro_beta = dro_beta
                self.dro_update_flag = True
            if self.venus_alpha is None and venus_alpha is not None and 'v' in keypoints:
                print('6666666666666')
                self.venus_alpha = venus_alpha
                self.venus_beta = venus_beta
                self.venus_update_flag = True


            delta_t = self.time_measure[1] - self.time_measure[0]
            self.time_measure.popleft()
            # 状态估计协方差矩阵P初始化（其实就是初始化最优解的协方差矩阵）
            P = np.eye(6)
            P[0:3, 0:3] = np.eye(3) * P_xyz_cov ** 2
            P[3:6, 3:6] = np.eye(3) * P_Vxyz_cov ** 2
            self.P_posterior = np.array(P)  # P_posterior是继续更新最优解的协方差矩阵
            X_prior, F, moon_pos, sun_pos = Cislunar_Update6_state(self.X_posterior, delta_t, self.time_measure[0])
            '''
            获取此时此刻金星、L4、L5、DRO、月球到地球的位置
            '''
            def alpha_beta_posterior(position):
                alpha = math.atan2(position[1], position[0])
                beta = math.atan2(position[2], math.sqrt(position[0] ** 2 + position[1] ** 2))
                return alpha, beta
            position_l42e, lighttime = spice.spkpos('Venus', timestamps, 'J2000', 'NONE', 'Earth')
            position_l52e, lighttime = spice.spkpos('Venus', timestamps, 'J2000', 'NONE', 'Earth')
            position_dro2e, lighttime = spice.spkpos('Venus', timestamps, 'J2000', 'NONE', 'Earth')
            position_venus2e, lighttime = spice.spkpos('VENUS', timestamps, 'J2000', 'NONE', 'Earth')
            '''
            在这个时候就拉齐观测量
            '''
            m_rho_rk = np.linalg.norm(moon_pos - X_prior[0:3])
            m_alpha_rk, m_beta_rk = alpha_beta_posterior(moon_pos - X_prior[0:3])
            e_alpha_rk, e_beta_rk = alpha_beta_posterior(-X_prior)
            l4_alpha_rk, l4_beta_rk = alpha_beta_posterior(position_l42e*1000 - X_prior[0:3])
            l5_alpha_rk, l5_beta_rk = alpha_beta_posterior(position_l52e*1000 - X_prior[0:3])
            dro_alpha_rk, dro_beta_rk = alpha_beta_posterior(position_dro2e*1000 - X_prior[0:3])
            venus_alpha_rk, venus_beta_rk = alpha_beta_posterior(position_venus2e*1000 - X_prior[0:3])

            # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
            Q = np.eye(6)
            Q[0:3, 0:3] = np.eye(3) * Q_xyz_cov ** 2
            Q[3:6, 3:6] = np.eye(3) * Q_Vxyz_cov ** 2
            # 计算状态估计协方差矩阵P
            P_prior_1 = np.dot(F, self.P_posterior)  # P_posterior是上一时刻最优估计的协方差矩阵    # P_prior_1就为公式中的（F.Pk-1）
            P_prior = np.dot(P_prior_1, F.T) + Q  # P_prior是得出当前的先验估计协方差矩阵      # Q是过程协方差
            # ------------------- R|K|H|Z更新  ------------------------
            [Px, Py, Pz] = X_prior[0:3]
            [Xm, Ym, Zm] = moon_pos
            [Xl4, Yl4, Zl4] = position_l42e*1000
            [Xl5, Yl5, Zl5] = position_l52e*1000
            [Xdro, Ydro, Zdro] = position_dro2e*1000
            [Xvenus, Yvenus, Zvenus] = position_venus2e*1000

            # 避免被除数为0
            position_rho_m = sqrt((Xm - Px) ** 2 + (Ym - Py) ** 2 + (Zm - Pz) ** 2)
            if position_rho_m < 1e-8:
                position_rho_m = 1e-8
            # 线性化(将非线性转为线性) 测量的协方差矩阵R，一般厂家给提供，R中的值越小，说明测量的越准确。
            # Z_X, Z_measure = None, None
            if self.mode == 1:
                # 观测所有
                R = np.eye(len(keypoints) * 2 + 1) * m2c_err ** 2
                R[0, 0] = rho_err ** 2
                H_old = [ #月球观测量敏感度
                     np.array([[(Px - Xm) / position_rho_m, (Py - Ym) / position_rho_m, (Pz - Zm) / position_rho_m, 0, 0, 0],
                     [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),-(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                     [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2), 0, 0, 0]]).squeeze() if 'm' in keypoints else None,
                     # 地球观测量敏感度
                     np.array([[-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                     [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)), Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                      -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]]).squeeze() if 'e' in keypoints else None,
                     # l4 观测量测量敏感度
                        np.array([[-(Py - Yl4) / ((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2),
                         -(-Px + Xl4) / ((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2), 0, 0, 0, 0],
                        [(Px - Xl4) * (Pz - Zl4) / (sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) * (
                                    (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2)),
                         (Py - Yl4) * (Pz - Zl4) / (sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) * (
                                     (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2)),
                         -sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) / (
                                     (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2), 0, 0, 0]]).squeeze() if '4' in keypoints else None,
                        # l5 观测量测量敏感度
                        np.array([[-(Py - Yl5) / ((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2),
                         -(-Px + Xl5) / ((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2), 0, 0, 0, 0],
                        [(Px - Xl5) * (Pz - Zl5) / (sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) * (
                                (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2)),
                         (Py - Yl5) * (Pz - Zl5) / (sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) * (
                                 (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2)),
                         -sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) / (
                                 (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2), 0, 0, 0]]).squeeze() if '5' in keypoints else None,
                        # dro 观测量测量敏感度
                        np.array([[-(Py - Ydro) / ((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2),
                         -(-Px + Xdro) / ((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2), 0, 0, 0, 0],
                        [(Px - Xdro) * (Pz - Zdro) / (sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) * (
                                (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2)),
                         (Py - Ydro) * (Pz - Zdro) / (sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) * (
                                 (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2)),
                         -sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) / (
                                 (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2), 0, 0, 0]]).squeeze() if 'd' in keypoints else None,
                        # venus 观测量测量敏感度
                        np.array([[-(Py - Yvenus) / ((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2),
                         -(-Px + Xvenus) / ((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2), 0, 0, 0, 0],
                        [(Px - Xvenus) * (Pz - Zvenus) / (sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) * (
                                (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2)),
                         (Py - Yvenus) * (Pz - Zvenus) / (sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) * (
                                 (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2)),
                         -sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) / (
                                 (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2), 0, 0, 0]]).squeeze() if 'v' in keypoints else None]
                Z_X_old = [np.array([m_rho_rk, m_alpha_rk, m_beta_rk]).squeeze() if 'm' in keypoints else None,
                                np.array([e_alpha_rk, e_beta_rk]).squeeze() if 'e' in keypoints else None,
                                np.array([l4_alpha_rk, l4_beta_rk]).squeeze() if '4' in keypoints else None,
                                np.array([l5_alpha_rk,l5_beta_rk]).squeeze() if '5' in keypoints else None,
                                np.array([dro_alpha_rk,dro_beta_rk]).squeeze() if 'd' in keypoints else None,
                                np.array([venus_alpha_rk,venus_beta_rk]).squeeze() if 'v' in keypoints else None]
                Z_measure_old = [np.array([self.m_rho, self.m_alpha, self.m_beta]).squeeze() if 'm' in keypoints else None,
                                      np.array([self.e_alpha, self.e_beta]).squeeze() if 'e' in keypoints else None,
                                      np.array([self.l4_alpha,self.l4_beta]).squeeze() if '4' in keypoints else None,
                                      np.array([self.l5_alpha,self.l5_beta]).squeeze() if '5' in keypoints else None,
                                      np.array([self.dro_alpha,self.dro_beta]).squeeze() if 'd' in keypoints else None,
                                      np.array([self.venus_alpha,self.venus_beta]).squeeze() if 'v' in keypoints else None]
                # H = np.array(list(filter(lambda x: x is not None, H)))
                # Z_X = np.array(list(filter(lambda x: x is not None, Z_X)))
                # Z_measure = np.array(list(filter(lambda x: x is not None, Z_measure)))
                H = []
                Z_X = []
                Z_measure = []
                for obv,zx,zm in zip(H_old,Z_X_old,Z_measure_old):
                    if obv is not None:
                        for obv_ in obv:
                            H.append(obv_)
                        for zx_ in zx:
                            Z_X.append(zx_)
                        for zm_ in zm:
                            Z_measure.append(zm_)
                H = np.array(H)
                Z_X = np.array(Z_X)
                Z_measure = np.array(Z_measure)
                # print(H)
                # print(Z_X)
                # print(Z_measure)
                # 计算卡尔曼增益
            else:
                if self.e_alpha is not None and self.m_alpha is not None:
                    # # 这个时候需要拉齐上一帧标记的观测量到这一帧
                    if self.flag_front == 'M' and self.time_front is not None and self.X_posterior is not None:
                        # 上一观测量是月球则更新对月球的观测量
                        X_prior_front, F_front, moon_pos_front, sun_pos_front = Cislunar_Update6_state(self.X_posterior,
                                                                                                       timestamps - self.time_front,
                                                                                                       self.time_front)
                        T_m2c = moon_pos_front -X_prior_front[0:3]
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

                    R = np.eye(5) * m2c_err ** 2
                    R[0, 0] = rho_err ** 2
                    H = np.array(
                        [[(Px - Xm) / position_rho_m, (Py - Ym) / position_rho_m, (Pz - Zm) / position_rho_m, 0, 0, 0],
                         [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                          -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                         [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                     (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2), 0, 0, 0],
                         [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                         [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                          -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                    Z_X = np.array([m_rho_rk, m_alpha_rk, m_beta_rk, e_alpha_rk, e_beta_rk])
                    Z_measure = np.array([self.m_rho, self.m_alpha, self.m_beta, self.e_alpha, self.e_beta])
                    self.flag_time_update = True # 当地月观测量都有的时候，可以进行时间更新，然后把上一帧标记的观测量置空，更新上一帧观测量为现在观测量
                    if self.flag_front == 'M':
                        self.flag_front_front = 'M'
                        self.flag_front = 'E'
                        self.time_front = timestamps # 这个时间戳会更新成最新进入的观测量的时间戳，然后成为下一个的上一观测量
                        self.m_rho = None
                        self.m_alpha = None
                        self.m_beta = None
                        print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMEEEEEEEEEEEEEEEEEEEEEEEEEEE')
                    else:
                        self.flag_front = 'M'
                        self.flag_front_front = 'E'
                        self.time_front = timestamps
                        self.e_rho = None
                        self.e_alpha = None
                        self.e_beta = None
                        print('EEEEEEEEEEEEEEEEEEEEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
                elif self.e_alpha is None:
                    self.time_front = timestamps
                    self.flag_front = 'M' # 如果持续是同一观测量，则更新同一观测量，时间仍然为标记上一帧为异测量量的时间戳
                    self.flag_front_front = 'M'
                    self.flag_time_update = False
                    print('***********************************')
                elif self.m_alpha is None:
                    self.time_front = timestamps
                    self.flag_front = 'E'
                    self.flag_front_front = 'E'
                    self.flag_time_update = False
                    print("/////////////////////////////////////////")

            if self.flag_time_update == True:
                k1 = np.dot(P_prior, H.T)  # P_prior是得出当前的先验估计协方差矩阵
                k2 = np.dot(np.dot(H, P_prior), H.T) + R  # R是测量的协方差矩阵
                K = np.dot(k1, np.linalg.inv(k2))  # np.linalg.inv()：矩阵求逆   # K就是当前时刻的卡尔曼增益
                # 测量值
                '''
                X_posterior_1 = Z_measure - np.dot(H, X_prior) X_prior表示根据上一时刻的最优估计值得到当前的估计值
                由于观测矩阵的非线性程度太高，因此不能利用H矩阵计算观测 # 用当前X_prior的位置更新残差后项，重新计算对应的观测值
                '''
                self.X_posterior_1 = Z_measure - Z_X
                # self.X_posterior_1 = Z_measure - np.dot(H, X_prior)
                self.X_posterior = X_prior + np.dot(K, self.X_posterior_1)  # X_posterior是根据估计值及当前时刻的观测值融合到一体得到的最优估计值
                # 最优估计值从m换成km
                self.position_x_posterior_est = self.X_posterior[0] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
                self.position_y_posterior_est = self.X_posterior[1] # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
                self.position_z_posterior_est = self.X_posterior[2]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z位置值存入到列表中
                self.speed_x_posterior_est = self.X_posterior[3]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
                self.speed_y_posterior_est = self.X_posterior[4]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中
                self.speed_z_posterior_est = self.X_posterior[5]  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z速度值存入到列表中
                # print(np.linalg.norm([self.position_x_posterior_est[i-2] - self.position_x_true[i-1],
                #                       self.position_y_posterior_est[i-2] - self.position_y_true[i-1],
                #                       self.position_z_posterior_est[i-2] - self.position_z_true[i-1]]))
                # 更新状态估计协方差矩阵P     （其实就是继续更新最优解的协方差矩阵）
                P_posterior_1 = np.eye(6) - np.dot(K, H)  # np.eye(4)返回一个4维数组，对角线上为1，其他地方为0，其实就是一个单位矩阵
                self.P_posterior = np.dot(P_posterior_1, P_prior)  # P_posterior是继续更新最优解的协方差矩阵  # P_prior是得出的当前的先验估计协方差矩阵
                X_posterior_1_str = ' '.join(str(i) for i in self.X_posterior_1)
                str_es_q = None
                if type(self.es_q) is np.ndarray:
                    str_es_q = ' '.join(str(i) for i in self.es_q)
                    str_es_q += ' '
                else:
                    str_es_q = ''
                self.log_file.write(str(self.time_measure[0]) + " " +
                                        str(self.position_x_posterior_est) + " " +
                                        str(self.position_y_posterior_est) + " " +
                                        str(self.position_z_posterior_est) + " " +
                                        str(self.speed_x_posterior_est) + " " +
                                        str(self.speed_y_posterior_est) + " " +
                                        str(self.speed_z_posterior_est) + " " +
                                        str(single_x) + " " +
                                        str(single_y) + " " +
                                        str(single_z) + " " +
                                        str(0) + " " +
                                        str(0) + " " +
                                        str_es_q +
                                        X_posterior_1_str + "\n")

    def ekf_update_batch(self,
                           case_H,
                           rho_err=2,
                           m2c_err=1e-10,
                           P_yxz_cov=1,
                           P_Vxyz_cov=1e-5,
                           Q_xyz_cov=1e-10,
                           Q_Vxyz_cov=1e-10,keypoints=None):
        # --------------------------- 初始化 -------------------------
        # 用第2帧测量数据初始化
        # mu = 0
        # sigma = 50
        # random.gauss(mu, sigma) / 5
        print(keypoints)
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
        X_posterior = np.array(X0)  # X_posterior表示上一时刻的最优估计值
        P_posterior = np.array(P)  # P_posterior是继续更新最优解的协方差矩阵
        # 将初始化后的数据依次送入(即从第三帧速度往里送)
        delta_t = 0
        len_ = len(self.time_measure)

        for i in range(2, len_):
            # ------------------- 下面开始进行预测和更新，来回不断的迭代 -------------------------
            # 求前后两帧的时间差，数据包中的时间戳单位为微秒，处以1e6，转换为秒
            # ---------------------- 时间更新  -------------------------
            delta_t = self.time_measure[i] - last_timestamp_
            Z_X = []
            # 先验估计值与状态转移矩阵
            X_prior, F, moon_pos, sun_pos = Cislunar_Update6_state(X_posterior, delta_t, last_timestamp_)
            self.moon_pos.append(moon_pos)
            X_m2c = moon_pos - X_prior[0:3]
            rho = np.linalg.norm(X_m2c)
            alpha = math.atan2(X_m2c[1], X_m2c[0])
            beta = math.atan2(X_m2c[2], math.sqrt(X_m2c[0] ** 2 + X_m2c[1] ** 2))

            rho_e = np.linalg.norm(X_prior[0:3])
            e_alpha = math.atan2(-X_prior[1], -X_prior[0])
            e_beta = math.atan2(-X_prior[2], np.sqrt(X_prior[0] ** 2 + X_prior[1] ** 2))

            X_s2c = sun_pos - X_prior[0:3]
            s_alpha = math.atan2(X_s2c[1], X_s2c[0])
            s_beta = math.atan2(X_s2c[2], sqrt(X_s2c[0] ** 2 + X_s2c[1] ** 2))

            venus_position,lighttime = spice.spkpos("Venus", self.time_measure[i], 'J2000', "NONE", "10002")
            v_alpha = math.atan2(venus_position[1],venus_position[0])
            v_beta = math.atan2(venus_position[2],sqrt(venus_position[0] ** 2 + venus_position[1] ** 2))

            l4_position, lighttime = spice.spkpos("Venus", self.time_measure[i], 'J2000', "NONE", "10002")
            l4_alpha = math.atan2(l4_position[1],l4_position[0])
            l4_beta = math.atan2(l4_position[2],sqrt(l4_position[0] ** 2 + l4_position[1] ** 2))

            l5_position, lighttime = spice.spkpos("Venus", self.time_measure[i], 'J2000', "NONE", "10002")
            l5_alpha = math.atan2(l5_position[1],l5_position[0])
            l5_beta = math.atan2(l5_position[2],sqrt(l5_position[0] ** 2 + l5_position[1] ** 2))

            dro_position, lighttime = spice.spkpos("Venus", self.time_measure[i], 'J2000', "NONE", "10002")
            dro_alpha = math.atan2(dro_position[1],dro_position[0])
            dro_beta = math.atan2(dro_position[2],sqrt(dro_position[0] ** 2 + dro_position[1] ** 2))


            last_timestamp_ = self.time_measure[i]
            self.position_x_prior_est.append(X_prior[0])  # 将根据上一时刻计算得到的x方向最优估计位置值添加到列表position_x_prior_est中
            self.position_y_prior_est.append(X_prior[1])  # 将根据上一时刻计算得到的y方向最优估计位置值添加到列表position_y_prior_est中
            self.position_z_prior_est.append(X_prior[2])  # 将根据上一时刻计算得到的y方向最优估计位置值添加到列表position_z_prior_est中
            self.speed_x_prior_est.append(X_prior[3])  # 将根据上一时刻计算得到的x方向最优估计速度值添加到列表speed_x_prior_est中
            self.speed_y_prior_est.append(X_prior[4])  # 将根据上一时刻计算得到的x方向最优估计速度值添加到列表speed_y_prior_est中
            self.speed_z_prior_est.append(X_prior[5])  # 将根据上一时刻计算得到的x方向最优估计速度值添加到列表speed_z_prior_est中
            # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
            Q = np.eye(6)
            Q[0:3, 0:3] = np.eye(3) * Q_xyz_cov ** 2
            Q[3:6, 3:6] = np.eye(3) * Q_Vxyz_cov ** 2
            # 计算状态估计协方差矩阵P
            P_prior_1 = np.dot(F, P_posterior)  # P_posterior是上一时刻最优估计的协方差矩阵    # P_prior_1就为公式中的（F.Pk-1）
            P_prior = np.dot(P_prior_1, F.T) + Q  # P_prior是得出当前的先验估计协方差矩阵      # Q是过程协方差
            # ------------------- R|K|H|Z更新  ------------------------
            [Px, Py, Pz] = X_prior[0:3]
            [Xm, Ym, Zm] = moon_pos
            # 避免被除数为0
            position_rho_ = sqrt((Xm - Px) ** 2 + (Ym - Py) ** 2 + (Zm - Pz) ** 2)

            if position_rho_ < 1e-8:
                position_rho_ = 1e-8
            # 线性化(将非线性转为线性) 测量的协方差矩阵R，一般厂家给提供，R中的值越小，说明测量的越准确。
            if case_H == 1:  # cace1: 仅对月球测距
                R = np.eye(1) * rho_err ** 2
                H = self.H_moondis(Px, Py, Pz, moon_pos, position_rho_)
                Z_X = np.array([rho])
                Z_measure = np.array([self.position_rho_measure[i]])
            if case_H == 2:  # case2: 对月球测距、转系后的赤经赤纬(方位角、俯仰角)
                R = np.eye(3) * m2c_err ** 2
                R[0, 0] = rho_err ** 2
                H = np.array(
                    [[(Px - Xm) / position_rho_, (Py - Ym) / position_rho_, (Pz - Zm) / position_rho_, 0, 0, 0],
                     [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                      -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                      0, 0, 0, 0],
                     [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                             (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                              (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                      0,
                      0, 0]])
                Z_X = np.array([rho, alpha, beta])
                Z_measure = np.array([self.position_rho_measure[i],
                                      self.position_a_measure[i],
                                      self.position_b_measure[i]])
            if case_H == 3:  # case3: 对月球测距、转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角
                R = np.eye(5) * m2c_err ** 2
                R[0, 0] = rho_err ** 2
                # R[2, 2] = rho_err ** 2
                H = np.array(
                    [[(Px - Xm) / position_rho_, (Py - Ym) / position_rho_, (Pz - Zm) / position_rho_, 0, 0, 0],
                     [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                      -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                      0, 0, 0, 0],
                     [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                             (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                              (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                      -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                      0,
                      0, 0],
                     # [Px / rho_e, Py / rho_e, Pz / rho_e, 0, 0, 0],
                     [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                     [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                      Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                      -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                Z_X = np.array([rho, alpha, beta, e_alpha, e_beta])
                Z_measure = np.array([self.position_rho_measure[i],
                                      self.position_a_measure[i],
                                      self.position_b_measure[i],
                                      # np.linalg.norm([self.position_x_measure[i],self.position_y_measure[i],self.position_z_measure[i]]),
                                      self.e_alpha_list[i],
                                      self.e_beta_list[i]])
            if case_H == 4:  # case4: 对月球转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角
                R = np.eye(4) * m2c_err ** 2
                H = np.array([[-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                       (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                           (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                               0, 0, 0],
                              [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                              [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0]])
                Z_X = np.array([alpha, beta, e_alpha, e_beta])
                Z_measure = np.array([self.position_a_measure[i],
                                      self.position_b_measure[i],
                                      self.e_alpha_list[i],
                                      self.e_beta_list[i]])
            if case_H == 5:  # case5: 对月球、地球、太阳测方位角与俯仰角
                Xs = sun_pos[0]
                Ys = sun_pos[1]
                Zs = sun_pos[2]
                R = np.eye(6) * m2c_err ** 2
                H = np.array([[-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                       (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                       (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                               0, 0, 0],
                              [-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                              [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0],
                              [-(Py - Ys) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               -(-Px + Xs) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xs) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                      (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               (Py - Ys) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               -sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) / (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2),
                               0, 0, 0]])
                Z_X = np.array([alpha, beta, e_alpha, e_beta, s_alpha, s_beta])
                Z_measure = np.array([self.position_a_measure[i],
                                      self.position_b_measure[i],
                                      self.e_alpha_list[i],
                                      self.e_beta_list[i],
                                      self.s_alpha_list[i],
                                      self.s_beta_list[i]])
            if case_H == 6:  # 对地球、太阳测角
                Xs = sun_pos[0]
                Ys = sun_pos[1]
                Zs = sun_pos[2]
                R = np.eye(4) * m2c_err ** 2
                H = np.array([[-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                              [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0, 0],
                              [-(Py - Ys) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               -(-Px + Xs) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xs) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                      (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               (Py - Ys) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               -sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) / (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2),
                               0, 0, 0]])
                Z_X = np.array([e_alpha, e_beta, s_alpha, s_beta])
                Z_measure = np.array([self.e_alpha_list[i],
                                      self.e_beta_list[i],
                                      self.s_alpha_list[i],
                                      self.s_beta_list[i]])
            if case_H == 7:  # 对月球、太阳测角
                Xs = sun_pos[0]
                Ys = sun_pos[1]
                Zs = sun_pos[2]
                R = np.eye(4) * m2c_err ** 2
                H = np.array([[-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                       (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                               -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                       (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2),
                               0, 0, 0],
                              [-(Py - Ys) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               -(-Px + Xs) / ((-Px + Xs) ** 2 + (-Py + Ys) ** 2),
                               0, 0, 0, 0],
                              [(Px - Xs) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                      (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               (Py - Ys) * (Pz - Zs) / (sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) * (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2)),
                               -sqrt((-Px + Xs) ** 2 + (-Py + Ys) ** 2) / (
                                       (-Px + Xs) ** 2 + (-Py + Ys) ** 2 + (-Pz + Zs) ** 2),
                               0, 0, 0]])
                Z_X = np.array([alpha, beta, s_alpha, s_beta])
                Z_measure = np.array([self.position_a_measure[i],
                                      self.position_b_measure[i],
                                      self.s_alpha_list[i],
                                      self.s_beta_list[i]])
            if case_H == 8:  # case8: 对地月金星
                Xvenus = venus_position[0] * 1000
                Yvenus = venus_position[1] * 1000
                Zvenus = venus_position[2] * 1000
                Xl4 = l4_position[0] * 1000
                Yl4 = l4_position[1] * 1000
                Zl4 = l4_position[2] * 1000
                Xl5 = l5_position[0] * 1000
                Yl5 = l5_position[1] * 1000
                Zl5 = l5_position[2] * 1000
                Xdro = dro_position[0] * 1000
                Ydro = dro_position[1] * 1000
                Zdro = dro_position[2] * 1000

                R = np.eye(len(keypoints) * 2 + 1) * m2c_err ** 2
                R[0, 0] = rho_err ** 2
                H_old = [  # 月球观测量敏感度
                    np.array(
                        [[(Px - Xm) / position_rho_, (Py - Ym) / position_rho_, (Pz - Zm) / position_rho_, 0, 0, 0],
                         [-(Py - Ym) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2),
                          -(-Px + Xm) / ((-Px + Xm) ** 2 + (-Py + Ym) ** 2), 0, 0, 0, 0],
                         [(Px - Xm) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                     (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          (Py - Ym) * (Pz - Zm) / (sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) * (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2)),
                          -sqrt((-Px + Xm) ** 2 + (-Py + Ym) ** 2) / (
                                      (-Px + Xm) ** 2 + (-Py + Ym) ** 2 + (-Pz + Zm) ** 2), 0, 0,
                          0]]).squeeze() if 'm' in keypoints else None,
                    # 地球观测量敏感度
                    np.array([[-Py / (Px ** 2 + Py ** 2), Px / (Px ** 2 + Py ** 2), 0, 0, 0, 0],
                              [Px * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               Py * Pz / (sqrt(Px ** 2 + Py ** 2) * (Px ** 2 + Py ** 2 + Pz ** 2)),
                               -sqrt(Px ** 2 + Py ** 2) / (Px ** 2 + Py ** 2 + Pz ** 2), 0, 0,
                               0]]).squeeze() if 'e' in keypoints else None,
                    # l4 观测量测量敏感度
                    np.array([[-(Py - Yl4) / ((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2),
                               -(-Px + Xl4) / ((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2), 0, 0, 0, 0],
                              [(Px - Xl4) * (Pz - Zl4) / (sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) * (
                                      (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2)),
                               (Py - Yl4) * (Pz - Zl4) / (sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) * (
                                       (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2)),
                               -sqrt((-Px + Xl4) ** 2 + (-Py + Yl4) ** 2) / (
                                       (-Px + Xl4) ** 2 + (-Py + Yl4) ** 2 + (-Pz + Zl4) ** 2), 0, 0,
                               0]]).squeeze() if '4' in keypoints else None,
                    # l5 观测量测量敏感度
                    np.array([[-(Py - Yl5) / ((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2),
                               -(-Px + Xl5) / ((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2), 0, 0, 0, 0],
                              [(Px - Xl5) * (Pz - Zl5) / (sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) * (
                                      (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2)),
                               (Py - Yl5) * (Pz - Zl5) / (sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) * (
                                       (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2)),
                               -sqrt((-Px + Xl5) ** 2 + (-Py + Yl5) ** 2) / (
                                       (-Px + Xl5) ** 2 + (-Py + Yl5) ** 2 + (-Pz + Zl5) ** 2), 0, 0,
                               0]]).squeeze() if '5' in keypoints else None,
                    # dro 观测量测量敏感度
                    np.array([[-(Py - Ydro) / ((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2),
                               -(-Px + Xdro) / ((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2), 0, 0, 0, 0],
                              [(Px - Xdro) * (Pz - Zdro) / (sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) * (
                                      (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2)),
                               (Py - Ydro) * (Pz - Zdro) / (sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) * (
                                       (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2)),
                               -sqrt((-Px + Xdro) ** 2 + (-Py + Ydro) ** 2) / (
                                       (-Px + Xdro) ** 2 + (-Py + Ydro) ** 2 + (-Pz + Zdro) ** 2), 0, 0,
                               0]]).squeeze() if 'd' in keypoints else None,
                    # venus 观测量测量敏感度
                    np.array([[-(Py - Yvenus) / ((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2),
                               -(-Px + Xvenus) / ((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2), 0, 0, 0, 0],
                              [(Px - Xvenus) * (Pz - Zvenus) / (sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) * (
                                      (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2)),
                               (Py - Yvenus) * (Pz - Zvenus) / (sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) * (
                                       (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2)),
                               -sqrt((-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2) / (
                                       (-Px + Xvenus) ** 2 + (-Py + Yvenus) ** 2 + (-Pz + Zvenus) ** 2), 0, 0,
                               0]]).squeeze() if 'v' in keypoints else None]
                Z_X_old = [np.array([rho, alpha, beta]).squeeze() if 'm' in keypoints else None,
                           np.array([e_alpha, e_beta]).squeeze() if 'e' in keypoints else None,
                           np.array([l4_alpha, l4_beta]).squeeze() if '4' in keypoints else None,
                           np.array([l5_alpha, l5_beta]).squeeze() if '5' in keypoints else None,
                           np.array([dro_alpha, dro_beta]).squeeze() if 'd' in keypoints else None,
                           np.array([v_alpha, v_beta]).squeeze() if 'v' in keypoints else None]
                Z_measure_old = [
                    np.array([self.position_rho_measure[i], self.position_a_measure[i], self.position_b_measure[i]]).squeeze() if 'm' in keypoints else None,
                    np.array([self.e_alpha_list[i], self.e_beta_list[i]]).squeeze() if 'e' in keypoints else None,
                    np.array([self.l4_alpha_list[i], self.l4_beta_list[i]]).squeeze() if '4' in keypoints else None,
                    np.array([self.l5_alpha_list[i], self.l5_beta_list[i]]).squeeze() if '5' in keypoints else None,
                    np.array([self.dro_alpha_list[i], self.dro_beta_list[i]]).squeeze() if 'd' in keypoints else None,
                    np.array([self.v_alpha_list[i], self.v_beta_list[i]]).squeeze() if 'v' in keypoints else None]
                # H = np.array(list(filter(lambda x: x is not None, H)))
                # Z_X = np.array(list(filter(lambda x: x is not None, Z_X)))
                # Z_measure = np.array(list(filter(lambda x: x is not None, Z_measure)))
                H = []
                Z_X = []
                Z_measure = []
                for obv, zx, zm in zip(H_old, Z_X_old, Z_measure_old):
                    if obv is not None:
                        for obv_ in obv:
                            H.append(obv_)
                        for zx_ in zx:
                            Z_X.append(zx_)
                        for zm_ in zm:
                            Z_measure.append(zm_)
                H = np.array(H)
                Z_X = np.array(Z_X)
                Z_measure = np.array(Z_measure)
            # 计算卡尔曼增益
            k1 = np.dot(P_prior, H.T)  # P_prior是得出当前的先验估计协方差矩阵
            k2 = np.dot(np.dot(H, P_prior), H.T) + R  # R是测量的协方差矩阵
            K = np.dot(k1, np.linalg.inv(k2))  # np.linalg.inv()：矩阵求逆   # K就是当前时刻的卡尔曼增益
            # 测量值
            '''
            X_posterior_1 = Z_measure - np.dot(H, X_prior) X_prior表示根据上一时刻的最优估计值得到当前的估计值
            由于观测矩阵的非线性程度太高，因此不能利用H矩阵计算观测 # 用当前X_prior的位置更新残差后项，重新计算对应的观测值
            '''
            X_posterior_1 = Z_measure - Z_X
            self.X_posterior_1_list.append(X_posterior_1)
            X_posterior = X_prior + np.dot(K, X_posterior_1)  # X_posterior是根据估计值及当前时刻的观测值融合到一体得到的最优估计值
            # 最优估计值从m换成km
            self.position_x_posterior_est.append(X_posterior[0])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
            self.position_y_posterior_est.append(X_posterior[1])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
            self.position_z_posterior_est.append(X_posterior[2])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z位置值存入到列表中
            self.speed_x_posterior_est.append(X_posterior[3])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
            self.speed_y_posterior_est.append(X_posterior[4])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中
            self.speed_z_posterior_est.append(X_posterior[5])  # 根据估计值及当前时刻的观测值融合到一体得到的最优估计z速度值存入到列表中
            # print(np.linalg.norm([self.position_x_posterior_est[i-2] - self.position_x_true[i-1],
            #                       self.position_y_posterior_est[i-2] - self.position_y_true[i-1],
            #                       self.position_z_posterior_est[i-2] - self.position_z_true[i-1]]))
            # 更新状态估计协方差矩阵P     （其实就是继续更新最优解的协方差矩阵）
            P_posterior_1 = np.eye(6) - np.dot(K, H)  # np.eye(4)返回一个4维数组，对角线上为1，其他地方为0，其实就是一个单位矩阵
            P_posterior = np.dot(P_posterior_1, P_prior)  # P_posterior是继续更新最优解的协方差矩阵  # P_prior是得出的当前的先验估计协方差矩阵
            self.P_position_err.append(np.sqrt(np.array([P_posterior[0, 0]+P_posterior[1, 1]+P_posterior[2, 2]])))
            self.P_vec_err.append(np.sqrt(np.array([P_posterior[3, 3]+ P_posterior[4, 4]+ P_posterior[5, 5]])))
            cos_a = self.position_x_true[i]*1000 * X_posterior[0] + self.position_y_true[i]*1000 * X_posterior[1] + self.position_z_true[i]*1000 * X_posterior[2]
            cos_b = np.linalg.norm([self.position_x_true[i]*1000,self.position_y_true[i]*1000,self.position_z_true[i]*1000]) * np.linalg.norm(X_posterior[0:3])
            cos = cos_a / cos_b
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
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
            self.xyz_true_posterior_est()
            # 绘制单帧定位的位速误差图
            self.single_err_plot(case_H,
                                 rho_err,
                                 m2c_err,
                                 P_yxz_cov,
                                 P_Vxyz_cov,
                                 Q_xyz_cov,
                                 Q_Vxyz_cov,
                                 delta_t,
                                 self.time_measure[0],
                                 self.time_measure[len_ - 1],
                                 keypoints)
            # # 绘制角分辨率与测量量误差图
            self.Pointing_Precision_residual_plot()
            # 绘制测量量残差
            self.X_posterior_1_plot(case_H)
            # 绘制3d的xyz图
            self.threed_xyz()

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
                ax.plot(self.position_x_posterior_est[:i], self.position_y_posterior_est[:i], self.position_z_posterior_est[:i],label="ekf", c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.plot(self.moon_pos[:i,0], self.moon_pos[:i,1], self.moon_pos[:i,2],label="moon", c='c')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.scatter(0, 0, 0, label="earth", c='b', s=100, marker="o")
                ax.scatter(self.position_x_posterior_est[i], self.position_y_posterior_est[i], self.position_z_posterior_est[i], label="ekf_update",
                           c='y', s=100, marker="^")  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.set_xlabel('X')  # 设置x坐标轴
                ax.set_ylabel('Y')  # 设置y坐标轴
                ax.set_zlabel('Z')  # 设置z坐标轴
                ax.legend()
                plt.pause(0.001)
            # if i < 30:
            #     plt.savefig(r'C:\Users\zhaoy\Desktop\1\{}.png'.format(str(i)),dpi=600)
            # else:
            #     if i % 20 == 0:
            #         plt.savefig(r'C:\Users\zhaoy\Desktop\1\{}.png'.format(str(i)), dpi=600)
            if i == len(self.position_x_true) - 3:
                plt.show()
    def xyz_true_posterior_est(self):
        # 绘制真值与最优估计值图
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs[0, 0].plot(self.position_x_true[2:], "-", label="位置x_实际值", linewidth=1)
        axs[0, 0].plot(self.position_x_posterior_est, "-", label="位置x_最优估计值", linewidth=1)
        # print(self.position_a_true)
        # print(self.position_a_measure)
        # print(self.position_b_true)
        # print(self.position_b_measure)
        # print(self.position_x_prior_est)
        # print(self.position_x_posterior_est)
        # axs[0, 0].plot(position_x_posterior_est, "-", label="位置x_扩展卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
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
    def single_err_plot(self,case_H,rho_err,
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
            case = self.case_H_map[case_H]
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
        v_err_threshold_id = int(10)
        err_px = median_filter(err_px,25)
        err_py = median_filter(err_py,20)
        err_pz = median_filter(err_pz,25)
        def rmse(predictions, targets):
            pt = median_filter(predictions[v_err_threshold_id:] - targets[v_err_threshold_id:],25)
            return np.sqrt((pt ** 2).mean())
        # print(self.speed_x_posterior_est)
        print("从第",v_err_threshold_id, "步收敛稳定————")
        print(self.case_H_map[case_H], "的RMSE:")
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
            str(rmse(np.array(self.speed_x_posterior_est),np.array(self.speed_x_true[2:]))) + "," +
            str(rmse(np.array(self.speed_y_posterior_est),np.array(self.speed_y_true[2:]))) + "," +
            str(rmse(np.array(self.speed_z_posterior_est),np.array(self.speed_z_true[2:]))) + "\n"
        )
        csv_save.close()
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs[0, 0].plot(np.log10(err_px[v_err_threshold_id:]), "-", label="误差位置x", linewidth=1)
        axs[0, 0].set_title("误差位置x")
        axs[0, 0].set_xlabel('k')
        axs[0, 0].legend()

        axs[0, 1].plot(np.log10(err_py[v_err_threshold_id:]), "-", label="误差位置y", linewidth=1)
        axs[0, 1].set_title("误差位置y")
        axs[0, 1].set_xlabel('k')
        axs[0, 1].legend()

        axs[0, 2].plot(np.log10(err_pz[v_err_threshold_id:]), "-", label="误差位置z", linewidth=1)
        axs[0, 2].set_title("误差位置z")
        axs[0, 2].set_xlabel('k')
        axs[0, 2].legend()

        axs[1, 0].plot(np.log10(err_vx[v_err_threshold_id:]), "-", label="误差速度x", linewidth=1)
        axs[1, 0].set_title("误差速度x")
        axs[1, 0].set_xlabel('k')
        axs[1, 0].legend()

        axs[1, 1].plot(np.log10(err_vy[v_err_threshold_id:]), "-", label="误差速度y", linewidth=1)
        axs[1, 1].set_title("误差速度y")
        axs[1, 1].set_xlabel('k')
        axs[1, 1].legend()

        axs[1, 2].plot(np.log10(err_vz[v_err_threshold_id:]), "-", label="误差速度z", linewidth=1)
        axs[1, 2].set_title("误差速度z")
        axs[1, 2].set_xlabel('k')
        axs[1, 2].legend()
        plt.show()
        return v_err_threshold_id
    def X_posterior_1_plot(self,case_H):
        '''
        先验残差图
        '''
        len_ = len(self.X_posterior_1_list[0])
        fig, axs = plt.subplots(len_, 1, figsize=(20, 15))
        for i in range(len_):
            a = math.floor(i/4)
            b = i % 4
            label_title = "测量量 "+str(i)+" 的残差"
            axs[i].plot(np.array(self.X_posterior_1_list)[:,i], "-", label=label_title, linewidth=1)
            axs[i].set_title(self.case_H_map[case_H])
            axs[i].set_xlabel('k')
            axs[i].legend()
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
        '''
        处理X轴刻度坐标将其转化成从0开始的sec、min or h
        '''
        time_list = np.array(self.time_measure)
        x_lim_sec = time_list - time_list[0]
        x_lim_min = x_lim_sec / 60
        x_lim_h = x_lim_min / 60
        x_lim_day = x_lim_h / 24
        x_lim = x_lim_day
        print(x_lim_sec[-1] - x_lim_sec[0])
        time_name = '时间[day]'
        colors = list(mcolors.TABLEAU_COLORS.keys())
        i =  0
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 坐标图像中显示中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(dpi=110,figsize=(12,11))
        for k,v in self.err_3axis_dict.items():
            i += 1
            # plt.plot(x_lim[2:],v/1000,label=k,color=mcolors.TABLEAU_COLORS[colors[i]])
            plt.plot(x_lim[2:],v/1000,color=mcolors.TABLEAU_COLORS[colors[i]])

        # plt.axhline(y=100, c='r', ls='-.', lw=1.5, label='y=100')
        plt.axhline(y=30, c='g', ls='-.', lw=1.5, label='y=30')
        plt.ylabel('位置误差[km]', fontsize=20)
        plt.xlabel(time_name,fontsize=20)
        plt.tick_params(axis='y', labelsize=16)
        plt.tick_params(axis='x', labelsize=16)
        plt.legend(prop={'size': 16})
        plt.title('导航精度', fontsize=20)
        plt.show()


if __name__ == '__main__':
    fpath = "simulation_data/simulation_data_30_e-10.txt"
    fpath2 = 'simulation_data/misimulation_data_2w_gt_30_e-10.txt'
    ekf = EKF_ours(True, fpath, fpath2)
    # cace1: 仅对月球测距
    # case2: 对月球测距、转系后的赤经赤纬(方位角、俯仰角)
    # case3: 对月球测距、转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角
    # case4: 对月球转系后的赤经赤纬(方位角、俯仰角)、对地球测的方位角、俯仰角
    # case5: 对地球、月球、太阳测方位角与俯仰角
    # case6: 对地球、太阳测角
    # case7: 对月球、太阳测角
    # case8: 仅对月球测角
    # ekf.ekf_update_batch(1)
    # ekf.init_list()
    # ekf.ekf_update_batch(2)
    # ekf.init_list()
    # ekf.ekf_update_batch(3)
    # ekf.init_list()
    # ekf.ekf_update_batch(4)
    # ekf.init_list()
    # ekf.ekf_update_batch(5)
    # ekf.init_list()
    # ekf.ekf_update_batch(6)
    # ekf.init_list()
    # ekf.ekf_update_batch(7)
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m')
    # ekf.init_list()
    '''
    仅针对DRO场景下对地月观测，
    多余的其他的不写，加入其他观测量的不写
    根据图像处理得来的观测数据进行仿真
    其最终收敛精度也最趋向于图像仿真得来的 
    '''
    ekf.ekf_update_batch(8, keypoints='em')
    ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m4')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='md')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='em4')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m45')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='mdv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='em45')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m4dv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m45d')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='em45d')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='em4dv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m45dv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='em45dv')
    # ekf.init_list()
    ekf.err_3axis_plot()
    # ekf.ekf_update_batch(8, keypoints='emdv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='emd')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='emv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='mv')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='md')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m4')
    # ekf.init_list()
    # ekf.ekf_update_batch(8, keypoints='m5')
    # ekf.init_list()
    print(ekf.position_x_measure[-1] - ekf.position_x_true[-1])
    print(ekf.position_y_measure[-1] - ekf.position_y_true[-1])
    print(ekf.position_z_measure[-1] - ekf.position_z_true[-1])
    # ekf2 = EKF_ours(
    #     r"C:\Users\zhaoy\PycharmProjects\EarthMoon\TestData\measurement\measurement2024-08-31.txt",
    #     r"C:\Users\zhaoy\PycharmProjects\EarthMoon\TestData\gt\gt2024-06-26.txt",
    #     True
    # )
    # ekf2.ekf_update()
    # ekf.symb_Jac_Fun()
    # print(ekf.position_x_posterior_est)