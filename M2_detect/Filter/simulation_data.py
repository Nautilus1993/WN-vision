# -*- coding: utf-8 -*-
# @Author  : Zhao Yutao
# @Time    : 2024/9/3 18:44
# @Function: 仿测量数据
# @mails: zhaoyutao22@mails.ucas.ac.cn
import math
import random
import spiceypy as spice
import numpy as np
from utils.utils import load_furnsh
# 加载Spice内核文件
spice.tkvrsn("TOOLKIT")
load_furnsh('../kernel')

def alpha_beta(position, atype):
    '''
    返回点目标和面目标两种类型的测角信息，添加噪声不同
    '''
    # 面目标测角精度
    angle_sigma_area = math.sqrt(0.00036639754179936305) / 3
    angle_sigma_point = math.sqrt(1.2213251393312098e-05) / 3
    alpha = None
    beta = None
    if atype == 0:  # 面目标
        alpha = math.atan2(position[1], position[0]) + random.gauss(0, angle_sigma_area)
        beta = math.atan2(position[2], math.sqrt(position[0] ** 2 + position[1] ** 2)) + random.gauss(0,
                                                                                                      angle_sigma_area)
    if atype == 1:  # 点目标
        alpha = math.atan2(position[1], position[0]) + random.gauss(0, angle_sigma_point)
        beta = math.atan2(position[2], math.sqrt(position[0] ** 2 + position[1] ** 2)) + random.gauss(0,
                                                                                                      angle_sigma_point)
    # print(alpha,beta)
    return alpha, beta

def simulation_data(filename,gt_moon_file,data_count):
    # 生成data_count条的数据
    file = open(filename, 'w')
    gt_moon_file = open(gt_moon_file,'w')
    e_file = open(r"simulation_data\DRO.txt",'w')
    et_init = spice.str2et("2024 Sep 1 00:00:00")

    for i in range(data_count):
        # 获取当前时刻相对于ECI的位置、速度
        timetamps = et_init +float(i*30)
        xyz_Vxyz, lighttimes = spice.spkezr("10002", timetamps, "J2000", "NONE", "Earth")
        # 获取当前时刻相对于月球J2000系下的位置、速度
        xyz_Vxyz_2moon, lighttimes_2moon = spice.spkezr("10002",timetamps, "J2000", "NONE", "MOON")
        # 获取当前时刻月球相对于ECI的位置、速度
        xyz_Vxyz_M2E, lighttimes_M2E = spice.spkezr("MOON",timetamps,'J2000',"NONE","EARTH")
        # 视线矢量  加上1m/s的速度噪声、Xyz分别按照48，60，10的噪声进行添加
        xyz_Vxyz_S2E, lighttimes_S2E = spice.spkezr("SUN",timetamps,'J2000',"NONE","10002")
        xyz_HYGstar = np.array([-0.355253515,-0.859991843,0.366330537])*3e16
        # 金星到目标位置
        xyz_Vxyz_Venus2TLI, lighttimes_E2DRO= spice.spkezr("Venus", timetamps, 'J2000', "NONE", "10002")
        # DRO到目标位置
        xyz_Vxyz_DRO2TLI, lighttimes_E2DRO = spice.spkezr("Venus", timetamps, 'J2000', "NONE", "10002")
        # 获取当前L4相对于TLI
        xyz_Vxyz_L42TLI, lighttimes_L42TLI = spice.spkezr("Venus", timetamps, 'J2000', "NONE", "10002")
        # 获取当前L5相对于TLI
        xyz_Vxyz_L52TLI, lighttimes_L52TLI = spice.spkezr("Venus", timetamps, 'J2000', "NONE", "10002")
        if i == 0:
            T_m2c = -np.array([xyz_Vxyz_2moon[0]+random.gauss(0,1),
                              xyz_Vxyz_2moon[1]+random.gauss(0,1),
                              xyz_Vxyz_2moon[2]+random.gauss(0,1),
                              xyz_Vxyz_2moon[3]+random.gauss(0,0.0000001),
                              xyz_Vxyz_2moon[4]+random.gauss(0,0.0000001),
                              xyz_Vxyz_2moon[5]+random.gauss(0,0.0000001)])
            T_e2c = -np.array([xyz_Vxyz[0]+random.gauss(0,1),
                                xyz_Vxyz[1]+random.gauss(0,1),
                                xyz_Vxyz[2]+random.gauss(0,1),
                                xyz_Vxyz[3]+random.gauss(0,0.0000001),
                                xyz_Vxyz[4]+random.gauss(0,0.0000001),
                                xyz_Vxyz[5]+random.gauss(0,0.0000001)])
        else:
            T_m2c = -np.array([xyz_Vxyz_2moon[0]+random.gauss(0,1),
                              xyz_Vxyz_2moon[1]+random.gauss(0,1),
                              xyz_Vxyz_2moon[2]+random.gauss(0,1),
                              xyz_Vxyz_2moon[3]+random.gauss(0,0.0000001),
                              xyz_Vxyz_2moon[4]+random.gauss(0,0.0000001),
                              xyz_Vxyz_2moon[5]+random.gauss(0,0.0000001)])
            T_e2c = -np.array([xyz_Vxyz[0]+random.gauss(0,1),
                                xyz_Vxyz[1]+random.gauss(0,1),
                                xyz_Vxyz[2]+random.gauss(0,1),
                                xyz_Vxyz[3]+random.gauss(0,0.0000001),
                                xyz_Vxyz[4]+random.gauss(0,0.0000001),
                                xyz_Vxyz[5]+random.gauss(0,0.0000001)])
        e_alpha = math.atan2(T_e2c[1],T_e2c[0])
        e_beta = math.atan2(T_e2c[2],np.sqrt(T_e2c[0]**2+T_e2c[1]**2))
        # 测量矢量
        T_star2cam = np.array([xyz_HYGstar[0] - xyz_Vxyz[0],
                               xyz_HYGstar[1] - xyz_Vxyz[1],
                               xyz_HYGstar[2] - xyz_Vxyz[2]])
        s_alpha = math.atan2(xyz_Vxyz_S2E[1],xyz_Vxyz_S2E[0])
        s_beta = math.atan2(xyz_Vxyz_S2E[2],np.sqrt(xyz_Vxyz_S2E[0]**2+xyz_Vxyz_S2E[1]**2))
        a_theta = math.atan2(T_m2c[1], T_m2c[0])
        b_theta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))

        angle_sigma_point = math.sqrt(1.2213251393312098e-10) / 3
        v_alpha = math.atan2(xyz_Vxyz_Venus2TLI[1],xyz_Vxyz_Venus2TLI[0])+ random.gauss(0,angle_sigma_point)
        v_beta = math.atan2(xyz_Vxyz_Venus2TLI[2],math.sqrt(xyz_Vxyz_Venus2TLI[0] ** 2 + xyz_Vxyz_Venus2TLI[1] ** 2))+ random.gauss(0,angle_sigma_point)
        l4_alpha, l4_beta = alpha_beta(xyz_Vxyz_L42TLI,atype=1)
        l5_alpha, l5_beta = alpha_beta(xyz_Vxyz_L52TLI,atype=1)
        dro_alpha, dro_beta = alpha_beta(xyz_Vxyz_DRO2TLI,atype=1)

        # 测量径向速度
        es_dis_v = np.linalg.norm(T_m2c[3:])
        # 测量径向距离
        es_dis = np.linalg.norm(T_m2c[:3])
        xyz_Vxyz_measured = xyz_Vxyz_M2E - T_m2c
        file.write(str(timetamps) + " " +  # 时间s
                   str(es_dis) + " " +  # 月球矢量距离km
                   str(a_theta) + " " +  # 月球视线方位角弧度rad
                   str(b_theta) + " " +  # 月球视线俯仰角弧度
                   str(es_dis_v) + " " +  # 径向速度
                   str(xyz_Vxyz[0]) + " " +  # ECI位置真值
                   str(xyz_Vxyz[1]) + " " +
                   str(xyz_Vxyz[2]) + " " +  # km
                   str(xyz_Vxyz[3]) + " " +
                   str(xyz_Vxyz[4]) + " " +
                   str(xyz_Vxyz[5]) + " " + "0 0 " +
                   str(xyz_Vxyz_measured[0]) + " " + # ECI位置估计值
                   str(xyz_Vxyz_measured[1]) + " " +
                   str(xyz_Vxyz_measured[2]) + " " +
                   str(xyz_Vxyz_measured[3]) + " " +
                   str(xyz_Vxyz_measured[4]) + " " +
                   str(xyz_Vxyz_measured[5]) + " " +
                   str(e_alpha)+" "+
                   str(e_beta)+" "+
                   str(s_alpha)+" "+
                   str(s_beta)+" "+
                   str(v_alpha)+" "+
                   str(v_beta)+" "+
                   str(l4_alpha) + " " +
                   str(l4_beta) + " " +
                   str(l5_alpha) + " " +
                   str(l5_beta) + " " +
                   str(dro_alpha) + " " +
                   str(dro_beta) + "\n")
        gt_moon_file.write(
            str(timetamps) + " " +  # 时间s
            str(xyz_Vxyz_2moon[0])+ " " +
            str(xyz_Vxyz_2moon[1]) + " " +
            str(xyz_Vxyz_2moon[2]) + " " +
            str(xyz_Vxyz_2moon[3]) + " " +
            str(xyz_Vxyz_2moon[4]) + " " +
            str(xyz_Vxyz_2moon[5]) + "\n"
        )
        e_file.write(str(timetamps) + " " +  # 时间s
            str(xyz_Vxyz[0])+ " " +
            str(xyz_Vxyz[1]) + " " +
            str(xyz_Vxyz[2]) + " " +
            str(xyz_Vxyz[3]) + " " +
            str(xyz_Vxyz[4]) + " " +
            str(xyz_Vxyz[5]) + "\n")
    file.close()

simulation_data('simulation_data/simulation_data_30_e-10.txt',
                'simulation_data/misimulation_data_2w_gt_30_e-10.txt',
                int(1440))
    # measurement_txt.write(str(timestamps) + " " +  # 时间s
    #                                   str(es_dis_list[i]) + " " +  # 距离km
    #                                   str(a_theta_list[i]) + " " +  # 弧度rad
    #                                   str(b_theta_list[i]) + " " +  # 弧度rad
    #                                   str((es_dis_list[i] - es_dis_list[i - 1]) / delta_t)  # 径向速度km/s
    #                                   + " " + x + " " + y + " " + z + " " +  # km
    #                                   str((float(x) - float((filename[i - 1][11]))) / delta_t) + " " +  # km/s
    #                                   str((float(y) - float((filename[i - 1][12]))) / delta_t) + " " +  # km/s
    #                                   str((float(z) - float((filename[i - 1][13]))) / delta_t) + " " +
    #                                   str(Pointing_Precision_list[i]) + " " +
    #                                   str(residual_list[i]) + " " +
    #                                   str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
    #                                   str((xyz[i][0] - xyz[i - 1][0] / delta_t)) + " " +
    #                                   str((xyz[i][1] - xyz[i - 1][1] / delta_t)) + " " +
    #                                   str((xyz[i][2] - xyz[i - 1][2] / delta_t)) +"\n")
