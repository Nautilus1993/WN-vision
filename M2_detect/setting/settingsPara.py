# -*- coding: utf-8 -*-
# @Author  : Zhao Yutao
# @Time    : 2024/8/6 22:58
# @Function: 配置文件
# @mails: zhaoyutao22@mails.ucas.ac.cn
import math

width_cam = None  # mm
height_cam = None  # mm
f_cam = None  # mm
fov_w = None
fov_h = None
Width_p = None  # pixel
Heigh_p = None  # pixel
CF = f_cam
Pixel_w = None
Pixel_h = None
fov = [fov_w, fov_h]

Earth_r = 6378.1366 # MAXQ仿真中为6378.1366km，实际为6371.004
Moon_r = 1737.40    # 1737.10,#MAXQ仿真中为1737.40km，实际为1737.10
h_IR = 0.0  # 红外高度，单位：km 相机仿真中把红外层去除，认为为0
h_IR_Re = Earth_r + h_IR  # 地球红外半径
q_bool = True  # 是否采用质心模型 # 姿态是否采用真值
draw_center = False
draw_star = True
core_draw_cal = False # 姿态与质心提取采用真值
es_dis2gt_dis = False # 距离采用真值
withdraw_r_method = 1  # {0: 椭圆模型直接法，1:yolo关键点法}
starmap_pattern = 2 # {1: 金字塔， 2：主星投票, 3:选两边定一边}
dis_compensation = 0 # 质心距离补偿，根据上天实际提取情况进行更改(工程经验，偏差消除)
testdata = r"data/20241126-20du/em20dro"  # 测试集路径
starcatalog = r"star_ephemeris/Star_Datasets/hyg_dro_ID.csv" # 星表目录
starpattern = r"star_ephemeris/Pattern/AngDis20_20_L_index_sorted.txt" # 角距文件
'''
网络模型配置
'''
modelfile = r"Net/best.pt" # 模型文件
resize = 640
'''
非线性滤波优化
'''
sat_id_str = '10002'# 卫星编号
ekf_mode = 2 # 1:交叉进入滤波器 2：拉齐观测量
'''
蒙卡实验，高斯分布平均为0，标准差为
'''
monteCalrlo = 100
measure_sita = 0.001
'''
以下为多个相机的配置
'''

cam_select = 2 # 相机配置选择
if cam_select == 1:
    width_cam = 17.64  # mm
    height_cam = 13.32  # mm
    f_cam = 50  # mm
    fov_w = 2 * math.atan2(width_cam / 2, f_cam) * 180 / math.pi
    fov_h = 2 * math.atan2(height_cam / 2, f_cam) * 180 / math.pi
    Width_p = 4656  # pixel
    Heigh_p = 3520  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]
    # K = [] 可以添加畸变系数进行图像矫正
    # Noisy_img = 0  # 图像噪声处理、通过转灰度图、减去平均灰度，提取高星等星点

if cam_select == 2:
    width_cam = 25  # mm
    height_cam = 25  # mm
    f_cam = 70.8910227452213691374302  # mm
    fov_w = 20
    fov_h = 20
    Width_p = 2048  # pixel
    Heigh_p = 2048  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]

if cam_select == 3:
    width_cam = 29.44  # mm
    height_cam = 29.44  # mm
    f_cam = 180  # mm
    fov_w = 2 * math.atan2(width_cam / 2, f_cam) * 180 / math.pi
    fov_h = 2 * math.atan2(height_cam / 2, f_cam) * 180 / math.pi
    Width_p = 6004  # pixel
    Heigh_p = 6004  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]

if cam_select == 4:
    width_cam = 29.44  # mm
    height_cam = 29.44  # mm
    f_cam = 180  # mm
    fov_w = 2 * math.atan2(width_cam / 2, f_cam) * 180 / math.pi
    fov_h = 2 * math.atan2(height_cam / 2, f_cam) * 180 / math.pi
    Width_p = 2048  # pixel
    Heigh_p = 2048  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]


if cam_select == 5:
    width_cam = 27.6184  # mm
    height_cam = 27.6184  # mm
    f_cam = 480  # mm
    fov_w = 2 * math.atan2(width_cam / 2, f_cam) * 180 / math.pi
    fov_h = 2 * math.atan2(height_cam / 2, f_cam) * 180 / math.pi
    Width_p = 2048  # pixel
    Heigh_p = 2048  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]

if cam_select == 6: # DROA
    width_cam = 11.264  # mm
    height_cam = 11.264  # mm
    f_cam = 217.871  # mm
    fov_w = 2 * math.atan2(width_cam / 2, f_cam) * 180 / math.pi
    fov_h = 2 * math.atan2(height_cam / 2, f_cam) * 180 / math.pi
    Width_p = 2048  # pixel
    Heigh_p = 2048  # pixel
    CF = f_cam
    Pixel_w = width_cam / Width_p
    Pixel_h = height_cam / Heigh_p
    fov = [fov_w, fov_h]