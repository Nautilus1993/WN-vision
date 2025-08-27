# -*- coding: utf-8 -*-
# @mails: zhaoyutao22@mails.ucas.ac.cn
# @Author  : Zhao Yutao
# @Time    : 2024/7/1 12:29
# @Function: 测试redis服务端发送图像
# @mails: zhaoyutao22@mails.ucas.ac.cn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import numpy as np
from ultralytics import YOLO
from Core_extraction import More_Core_extra
from Filter.EKF import EKF_ours

'''
    单星单敏感器航天器位姿估计框架：此框架主要应用于GEO或更高轨道，视场角20°，焦距79mm可见光相机对地球或月球进行观测
    1.  首先对地月椭圆轮廓提取，得出其中心点坐标Xc,Xy与其长短轴ab和旋转角theta
    2.0  将其传入星图识别模块：利用Sift或其他兴趣点提取方法对兴趣点检测，利用点在圆外公式判定兴趣点是否为星点，保留正常星点
    2.1 在星点提取模块中，引入高斯滤波对图像进行处理降噪，如果过多星点则会降低匹配速度
    2.2 筛选后的星点通过金字塔或者三角投票算法进行星图识别
    2.3 识别的星点ID为帧间匹配做准备，保留每颗星点的最近邻信息矩阵
    2.4 星图识别到的星点在相机坐标系转成与J2000系相同指向的坐标，与DRO_Star_Table中的星点坐标信息
        通过QUSET法进行旋转搜索，得到相机相对于J2000系下的外旋矩阵与四元数
    3.  地月识别通过CV2椭圆拟合算法，在1中已得到椭圆相关参数
    3.1 通过椭圆模型直接法提取地球质心，并通过RT齐次矩阵求逆得出相机相对于J2000系下的坐标
    3.2 观测月球则需要多一步转换，求出月球在当前时刻的J2000系历元，进行相加得出相机相对于J2000系下的坐标
    4.  对于地月识别出的位置与姿态，进行EKF或图优化算法进行优化
    4.1 先对位置速度6状态量进行优化，通过观测量相机距地心距离distance，赤经a_theta,赤纬b_theta，径向速度speed，4观测量
    4.2 设置F，Q，P，R矩阵，根据实际情况调整Q，P，R的大小
    4.3 求解雅可比矩阵中间量测矩阵，进行每次预测及更新
    5.  绘制残差、位置与速度的真值、测量值、滤波后值、及在3d情况下绘图
'''
import setting.settingsPara as para
def get_parser():
    parser = argparse.ArgumentParser(description='单敏感器解算位姿框架', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--testdata', type=str, default=para.testdata,
                        help='测试数据')
    parser.add_argument('--model', type=str, default=para.modelfile,
                        help='椭圆权重文件')
    parser.add_argument('--starcatalog', type=str, default=para.starcatalog,
                        help='星表文件')
    parser.add_argument('--starpattern', type=str, default=para.starpattern,
                        help='角距文件')
    parser.add_argument('--CF', type=float, default=para.CF,
                        help='相机焦距')
    parser.add_argument('--fov', type=tuple, default=para.fov,
                        help='相机视场')
    parser.add_argument('--earthr', type=float, default=para.Earth_r, #MAXQ仿真中为6378.1366km，百度为6371.004
                        help='地球半径')
    parser.add_argument('--moonr', type=float, default=para.Moon_r, #1737.10,#MAXQ仿真中为1737.40km，百度为1737.10
                        help='月球半径')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    data = None
    star_file = None
    A_M = None
    dict = None
    model = None
    if para.q_bool == False: # 不采用真值，则进行赋值
        data = np.loadtxt(open(para.starcatalog, "rb"), delimiter=",", skiprows=1,
                          usecols=[0, 3, 4, 5])  # 获取所有星点J2000下xyz数据
        # print(data)
        Ang_Dis_M_file = open(para.starpattern, "r")
        A_M = []
        dict = {}
        starid = 0
        for line in Ang_Dis_M_file.readlines():
            curLine = line.split(" ")[:-1]
            # curLine[0] = curLine[0].split("[")[1]
            # curLine[len(curLine) - 1] = curLine[len(curLine) - 1].split("]")[0]
            A_M.append(list(map(float, curLine)))
            temp_list = list(map(float, curLine))
            length = len(temp_list)
            dict_x = {}
            for i in range(0,length,2):
                dict_x[int(temp_list[i])] = float("%.2f" % temp_list[i+1])
            dict[starid] = dict_x
            starid += 1
        star_file = open("star_ephemeris/Pattern/star_pattern_file.txt", "a")
        print("星表文件已加载······")
        print("导航星库已加载······")
        # model = load_model("my_model.h5")
    model = YOLO(args.model)
    print("网络模型已加载······")
    efk_em = EKF_ours(batching=False)
    if para.withdraw_r_method == 1:
        efk_em.set_ekf_type('Net')
    else:
        efk_em.set_ekf_type('Ed')
    # socket从这里接入图像、时间戳、四元数、位置、速度信息，超参传入
    # 即将star_file的命名进行拆解，将后续每个函数中涉及对图像名操作的进行修改参数及读取方式
    More_Core_extra(data = data,
                    star_file = star_file,
                    A_M = A_M,
                    dict = dict,
                    model = model,
                    img_path = args.testdata,
                    camera_f = args.CF,
                    Earth_r = args.earthr,
                    Moon_r = args.moonr,
                    ekf_filter = efk_em)