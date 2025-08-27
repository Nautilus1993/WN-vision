# -*- coding: utf-8 -*-
# @Author  : Zhao Yutao
# @Time    : 2024/7/1 12:29
# @Function: 星图识别模块
# @mails: zhaoyutao22@mails.ucas.ac.cn
import os
import time
import cv2
import numpy as np
import math
import statistics

from star_ephemeris.Feature_extractor.FeaturePoint_Harris import harris_corner_detector
from star_ephemeris.utils.Ang_TwoAxis_Solution import quaternion2euler, euler2quaternion, quaternion2rot, rot2quaternion
from star_ephemeris.utils.quest import quest
import matplotlib.pyplot as plt
from star_ephemeris.Feature_extractor.ORB_des import ORB_Feature
import random
import setting.settingsPara as para

def random_selectstar(len_star):
    s = []
    while (len(s) < 5):
        x = random.randint(0, len_star-1)
        if x not in s:
            s.append(x)
    return s


def draw_starid(path, one_Cordinate_List,star_allid,data):
    '''
    画星点并且表示id
    '''
    print(star_allid)
    img = cv2.imread(path)
    dst = np.ones(img.shape, dtype=np.uint8)
    Addtext = dst.copy()
    for i in range(len(one_Cordinate_List)):
        # print(star_allid[i])
        # print(one_Cordinate_List[i])
        text = str(int(data[star_allid[i][1]-2][0]))
        cv2.putText(Addtext,
                    text,
                    (round(one_Cordinate_List[i,0] / para.Pixel_w + para.Width_p/2)-30, round(para.Heigh_p/2 - one_Cordinate_List[i,1] / para.Pixel_h) + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,(100,200,200),1)
        starcore_ = (one_Cordinate_List[i,0] / para.Pixel_w + para.Width_p/2, para.Heigh_p/2 - one_Cordinate_List[i,1] / para.Pixel_h), (15, 15), 0
        # print(starcore_)
        cv2.ellipse(dst, starcore_, (0, 255, 0), 1, 2)
    # cv2.imshow("star1", Addtext)
    # cv2.waitKey()
    dst = cv2.add(cv2.resize(dst, (1200, 1200)), cv2.resize(Addtext, (1200, 1200)))
    dst = cv2.add(dst,cv2.resize(img,(1200,1200)))
    cv2.imshow("star",dst)
    cv2.imwrite("data/star_id.png",dst)
    cv2.waitKey()

def tri_vote(star_file,A_M,dict,region_flag,star_id_mode):
    '''
    三角形主星投票机制
    '''
    star_id_mode, jizhunxing, region_flag = None,None,None
    for i in range(len(S_Matrix)):
        star_id_mode = -1
        if star_id_mode == -1:
            star_id = []
            if int((len(S_Matrix) + 1) / 2) < 8:
                len_s = int((len(S_Matrix) + 1) / 2)
            else:
                len_s = 8
            for s2 in range(1, len_s):
                s3 = s2 + int(len(S_Matrix) / 2) - 1
                # print(s2)
                if region_flag == False:
                    # star_l = tri_Ang(A_M, star_id_mode+len(A_M)+1, i, s2, s3)
                    star_l = tri_Ang_dict(dict, star_id_mode+len(A_M)+1, i, s2, s3)
                else: #后端帧间优化，缩小检索范围
                    # star_l = tri_Ang(A_M, star_id_mode+len(A_M)+1, i, s2, s3) #暂时不考虑天区问题还有bug未解决
                    star_l = tri_Ang_dict(dict, star_id_mode+len(A_M)+1, i, s2, s3)
                for j in range(len(star_l)):
                    star_id.append(star_l[j])
            if len(star_id) == 0:
                continue
            # star_file.write(str(star_id) + "\n")
            # print(star_id)
            star_id_mode = statistics.mode(star_id)
            star_count = 0
            for star_count_i in range(len(star_id)):
                if star_id_mode == star_id[star_count_i]:
                    star_count += 1
            if star_count == 1:
                continue
            elif star_count > 1:
                jizhunxing = i
                region_flag = True
                print("置信度为：" + str(star_count / len(star_id)))
                break
            else:
                region_flag = False

    return star_id_mode, jizhunxing, region_flag

def four_vote(S_Matrix, A_M, dict):
    '''
    金字塔方法调用
    '''
    jizhunxing,star_id_mode = None,None
    for ii in range(len(S_Matrix) - 4):
        # star_id_mode = four_Ang(A_M, ii, ii + 1, ii + 2, ii + 3)
        # star_id_mode = four_Ang_dict(dict, ii, ii + 1, ii + 2, ii + 3, ii + 4)
        # print(star_id_mode)
        random_starlist = random_selectstar(len(S_Matrix))
        # star_id_mode = four_Ang(A_M,
        #                         random_starlist[0],
        #                         random_starlist[1],
        #                         random_starlist[2],
        #                         random_starlist[3])
        star_id_mode = four_Ang_dict(dict,               #添加金字塔容错机制
                                     random_starlist[0],
                                     random_starlist[1],
                                     random_starlist[2],
                                     random_starlist[3],
                                     random_starlist[4])
        # star_id_mode = two2one_vote(dict,  # 添加金字塔容错机制
        #                              random_starlist[0],
        #                              random_starlist[1],
        #                              random_starlist[2],
        #                              random_starlist[3],
        #                              random_starlist[4])
        if star_id_mode != None:
            jizhunxing = random_starlist[0]
            break
    return jizhunxing, star_id_mode


def main_star(ellipse, star_file, path, A_M, dict, data, num,region_flag,star_id_mode):
    '''
    :param ellipse: 椭圆参数
    :param star_file:星历文件
    :param path:图像
    :param A_M:星表角距矩阵
    :param dict:星表hasp表
    :param data:
    :param num: num代表不同特征算法，1：金字塔算法，2：三角投票特征算法
    :param region_flag: 星图跟踪所用标记，区域搜索
    :param star_id_mode:主星id进行星图跟踪，区域搜索
    :return: star_camera_List, star_Cordinate_List, min_list, FP_camaxis,region_flag,star_id_mode
    '''
    global S_Matrix, threshold
    S_Matrix, min_list, FP_camaxis = Sift_ADMatrix_min(ellipse, path)  # sift提取特征点还需要优化，点信息提取过少
    # S_Matrix, min_list, FP_camaxis = ORB_ADMatrix(path)
    # S_Matrix = ORB_ADMatrix(path)#orb提取特征点还需要优化，点信息提取过多
    starttime = time.time()
    # 设置阈值
    threshold = 0.05
    # 基准星
    jizhunxing = 0
    # 读取自建星库表
    # print(A_M)
    # print(S_Matrix)
    if num == 1:
        # 金字塔特征算法
        jizhunxing,star_id_mode = four_vote(S_Matrix, A_M, dict)
    if num == 2:
        # 三角投票特征算法
        # print("*----------"+str(star_id_mode))
        if region_flag == False:
            star_id_mode, jizhunxing, region_flag = tri_vote(star_file, A_M, dict,region_flag,star_id_mode)
        else:
            #前端优化之：帧间进行最近邻匹配确定主星在像存储矩阵中id
            #后端再优化之：可以认为star_id_mode帧间不用改变，直接传递使用，然后根据帧间运动趋势来选取下一个star_id_mode
            star_id_mode, jizhunxing, region_flag = tri_vote(star_file, A_M, dict, region_flag, star_id_mode)
            # print("+++++++++++")
            # jizhunxing = 0
        if region_flag == False: # 经过投票后如果还未确定当下星点id，则初始化star_id_mode,jizhunxing=0,0
            jizhunxing, star_id_mode = four_vote(S_Matrix,A_M,dict)
        print("基准星（第一颗星）id为：" + str(star_id_mode))
    if num == 3:
        jizhunxing, star_id_mode = two2one_vote(S_Matrix, dict)
    if jizhunxing is None:
        return None, None, None, None, False, None
    star_allid = All_Starid(A_M, jizhunxing, star_id_mode)
    star_allid.append([jizhunxing, star_id_mode])
    print(star_allid)
    endtime = time.time()
    print("搜索花费（s）：：" + str(endtime - starttime))
    star_allid_l = []
    one_Cordinate_List = []
    # star_file.write("基准星" + str(jizhunxing) + "的id为：" + str(star_id_mode) + "\n")
    # star_file.write(str(star_allid) + "\n")
    # star_file.write("搜索花费：" + str(endtime - starttime) + "\n")
    # 对每一颗星都进行遍历得到最真实解
    # for star in range(len(S_Matrix)):
    #     xxxxxx=[]
    #     for s2 in range(1, int((len(S_Matrix) + 1) / 2)):
    #         s3 = s2 + int(len(S_Matrix) / 2) - 1
    #         # print(s2)
    #         star_l = tri_Ang(star, s2, s3)
    #         for j in range(len(star_l)):
    #             xxxxxx.append(star_l[j])
    #     print(xxxxxx)
    #     print(statistics.mode(xxxxxx))
    for k in range(len(star_allid)):
        star_allid_l.append(star_allid[k][1])
        one_Cordinate_List.append(FP_camaxis[star_allid[k][0]])

    star_Cordinate_List = []
    for j in range(len(star_allid)):
        star_Cordinate_List.append(data[star_allid_l[j]])

    star_Cordinate_List = np.array(star_Cordinate_List)
    one_Cordinate_List = np.array(one_Cordinate_List)
    star_camera_List = np.zeros((len(star_allid), 3))

    for i in range(len(one_Cordinate_List)):
        # 转换单位向量 ,还需要令相机系的交换一下，与UE系相同,X->Y,Y->Z,Z->X
        fenmu = math.sqrt(one_Cordinate_List[i, 0] ** 2 + one_Cordinate_List[i, 1] ** 2 + one_Cordinate_List[i, 2] ** 2)
        # star_camera_List[i, 1] = -one_Cordinate_List[i, 0] / fenmu
        # star_camera_List[i, 2] = one_Cordinate_List[i, 1] / fenmu
        # star_camera_List[i, 0] = one_Cordinate_List[i, 2] / fenmu
        star_camera_List[i, 0] = one_Cordinate_List[i, 2] / fenmu
        star_camera_List[i, 1] = -one_Cordinate_List[i, 0] / fenmu
        star_camera_List[i, 2] = one_Cordinate_List[i, 1] / fenmu
    # print(star_camera_List)
    # print(star_Cordinate_List)
    if para.draw_star == True:
        draw_starid(path,one_Cordinate_List,star_allid, data) #画出每个识别后星点的id及位置

    return star_camera_List, star_Cordinate_List, min_list, FP_camaxis,region_flag,star_id_mode


def ORB_ADMatrix(path):
    kp = ORB_Feature(path)  # 用orb获取特征点坐标
    # kp.sort()
    print(kp)
    # print(len(kp))
    # 得到关键点在成像面上的点，即相机坐标系
    global FP_camaxis
    FP_camaxis = []
    count = 0
    for n, i in enumerate(kp):
        if i not in kp[:n]:
            count += 1
            # print(i)
            # bili = 1+(4608/3456)**2
            # pixel_size = ((1.6/1.56*10)/math.sqrt(bili))/3456
            # FP_camaxis.append([(i[0] - 1024) * 25 / 2048, (1024 - i[1]) * 25 / 2048, 70.8910227452213691374302])
            bili = 1+(4576/3432)**2
            pixel_size = ((1.6/1.56*10)/math.sqrt(bili))/3432
            FP_camaxis.append([(i[0] - 2288) * pixel_size, (1716 - i[1]) * pixel_size, 25])
            # FP_camaxis.append([(i[0] - 1224) * 0.0032365, (1024 - i[1]) * 0.0032372, 8])
    # FP_camaxis = np.zeros((count,3))  # 构造n*n矩阵
    FP_camaxis = np.array(FP_camaxis)
    # print(FP_camaxis)
    # if count > 20:
    #     FP_camaxis = FP_camaxis[np.random.choice(FP_camaxis.shape[0], size=20, replace=False), :]

    # print(FP_camaxis)

    # 计算星图中星点间角距
    S_Matrix = np.zeros((len(FP_camaxis), len(FP_camaxis)))
    for i in range(len(FP_camaxis)):
        for j in range(i, len(FP_camaxis)):
            cos_a = FP_camaxis[i, 0] * FP_camaxis[j, 0] + FP_camaxis[i, 1] * FP_camaxis[j, 1] + FP_camaxis[i, 2] * \
                    FP_camaxis[j, 2]
            cos_b = math.sqrt(FP_camaxis[i, 0] ** 2 + FP_camaxis[i, 1] ** 2 + FP_camaxis[i, 2] ** 2) * math.sqrt(
                FP_camaxis[j, 0] ** 2 + FP_camaxis[j, 1] ** 2 + FP_camaxis[j, 2] ** 2)
            cos = cos_a / cos_b
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
            # print(abs(math.acos(cos) * 180/math.pi))
            ad = abs(math.acos(cos) * 180 / math.pi)
            S_Matrix[i, j] = float("%.3f" % ad)
            S_Matrix[j, i] = float("%.3f" % ad)

    min_list = []
    if len(S_Matrix) >= 7:
        for k in range(len(S_Matrix)):
            min_three = np.sort(S_Matrix[k])
            min_three = min_three[1:7]
            min_three[3] = list(S_Matrix[k]).index(min_three[0])
            min_three[4] = list(S_Matrix[k]).index(min_three[1])
            min_three[5] = list(S_Matrix[k]).index(min_three[2])
            min_list.append(min_three)
    else:
        for k in range(len(S_Matrix)):
            min_three = [0, 0, 0, 0, 0, 0]
            xxx_ = np.sort(S_Matrix[k])
            min_three[0:3] = xxx_[1:4]
            min_three[3] = list(S_Matrix[k]).index(min_three[0])
            min_three[4] = list(S_Matrix[k]).index(min_three[1])
            min_three[5] = list(S_Matrix[k]).index(min_three[2])
            min_list.append(min_three)
    # print(S_Matrix)
    return S_Matrix, np.array(min_list), FP_camaxis


def Sift_ADMatrix_min(ellipse, path):
    img = cv2.imread(path)
    (Xc, Yc), (ellipse_a, ellipse_b), angle_R = ellipse
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result = cv2.GaussianBlur(rgb, (11, 11), 3)
    result = cv2.GaussianBlur(rgb, (3, 3), 5)
    # 得到关键点图像坐标系坐标
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(result, None)  # 找到关键点
    img = cv2.drawKeypoints(result, kp, img)  # 绘制关键点
    # cv2.imshow("1",img)
    # kp = harris_corner_detector(path)
    kp = cv2.KeyPoint_convert(kp)  # 特征点坐标
    # kp = np.array([[1548,47],
    #                [370,86],
    #                [1235,135],
    #                [1700,147],
    #                [802,177],
    #                [215,439],
    #                [8,444],
    #                [1711,471],
    #                [1123,824],
    #                [970,950],
    #                [1215,1041],
    #                [1844,1160]
    #                ])
    # print(kp)
    # 得到关键点在成像面上的点，即相机坐标系
    FP_camaxis = []
    for n, i in enumerate(kp):
        if i not in kp[:n]:
            # pixel_size = 70.8910227452213691374302*math.tan(10)/1024
            # 将椭圆认为成圆，椭圆的中心坐标为圆心坐标，长轴为圆半径，在此圆外的兴趣点才被认为是星点
            # 运用点在圆外公式进行判定：
            if (i[0] - Xc)**2 + (i[1] - Yc)**2 > (ellipse_b/2)**2:
                FP_camaxis.append([(i[0] - para.Width_p/2) * para.Pixel_w, (para.Heigh_p/2 - i[1]) * para.Pixel_h, para.CF])
                # FP_camaxis.append([(i[0] - 1224) * 0.0032365, (1024 - i[1]) * 0.0032372, 8])

            # bili = 1+(4608/3456)**2
            # pixel_size = ((1.6/1.56*10)/math.sqrt(bili))/3456
            # FP_camaxis.append([(i[0] - 2304) * pixel_size, (1728 - i[1]) * pixel_size, 25])
    # FP_camaxis = np.zeros((count,3))  # 构造n*n矩阵
    # 如果图像中挑选大于15个星点，则随机挑选15个
    if len(FP_camaxis) > 40:
        FP_camaxis.clear()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = cv2.GaussianBlur(rgb, (11, 11), 10)  # 可以更改核大小
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(result, None)  # 找到关键点
        img = cv2.drawKeypoints(result, kp, img)  # 绘制关键点
        # cv2.imshow("1",img)
        kp = cv2.KeyPoint_convert(kp)  # 特征点坐标
        for n, i in enumerate(kp):
            if i not in kp[:n]:
                if (i[0] - Xc) ** 2 + (i[1] - Yc) ** 2 > (ellipse_b/2) ** 2:
                    FP_camaxis.append([(i[0] - para.Width_p/2) * para.Pixel_w, (para.Heigh_p/2 - i[1]) * para.Pixel_h, para.CF])
                    # FP_camaxis.append([(i[0] - 1024) * 25 / 2048, (1024 - i[1]) * 25 / 2048, 70.8910227452213691374302])
    # print(FP_camaxis)
    FP_camaxis = np.array(FP_camaxis)
    # if len(FP_camaxis) > 20:
    #     FP_camaxis = FP_camaxis[np.random.choice(FP_camaxis.shape[0], size=20, replace=False), :]
    # print(FP_camaxis)
    min_list = []
    # 计算星图中星点间角距
    S_Matrix = np.zeros((len(FP_camaxis), len(FP_camaxis)))
    for i in range(len(FP_camaxis)):
        for j in range(i, len(FP_camaxis)):
            cos_a = FP_camaxis[i, 0] * FP_camaxis[j, 0] + FP_camaxis[i, 1] * FP_camaxis[j, 1] + FP_camaxis[i, 2] * \
                    FP_camaxis[j, 2]
            cos_b = math.sqrt(FP_camaxis[i, 0] ** 2 + FP_camaxis[i, 1] ** 2 + FP_camaxis[i, 2] ** 2) * math.sqrt(
                FP_camaxis[j, 0] ** 2 + FP_camaxis[j, 1] ** 2 + FP_camaxis[j, 2] ** 2)
            cos = cos_a / cos_b
            if cos > 1:
                cos = 1
            elif cos < -1:
                cos = -1
            # print(abs(math.acos(cos) * 180/math.pi))
            ad = abs(math.acos(cos) * 180 / math.pi)
            S_Matrix[i, j] = float("%.1f" % ad)
            S_Matrix[j, i] = float("%.1f" % ad)
    # print(S_Matrix)
    if len(S_Matrix) >= 7:
        for k in range(len(S_Matrix)):
            min_three = np.sort(S_Matrix[k])
            min_three = min_three[1:7]
            min_three[3] = list(S_Matrix[k]).index(min_three[0])
            min_three[4] = list(S_Matrix[k]).index(min_three[1])
            min_three[5] = list(S_Matrix[k]).index(min_three[2])
            min_list.append(min_three)
    else:
        for k in range(len(S_Matrix)):
            min_three = [0, 0, 0, 0, 0, 0]
            xxx_ = np.sort(S_Matrix[k])
            try:
                min_three[0:3] = xxx_[1:4]
                min_three[3] = list(S_Matrix[k]).index(min_three[0])
                min_three[4] = list(S_Matrix[k]).index(min_three[1])
                min_three[5] = list(S_Matrix[k]).index(min_three[2])
                min_list.append(min_three)
            except Exception as e:
                print(e)
                min_three = None

    return S_Matrix, np.array(min_list), FP_camaxis


# 返回列表第int(A_M[i][j - 1])行中寻找int(A_M[i][k - 1])的角距值，验证三角
def search_index(index_1, index_2, A_M):
    for kkk in range(0, len(A_M[index_1]), 2):
        if int(A_M[index_1][kkk]) == int(index_2):
            return kkk + 1
def search_index_dict(index_1,index_2,dict):
        if int(index_2) in dict[index_1]:
            pass

def two2one_vote(S_Matrix, dict):
    '''
    选两边定一边
    '''
    star_id = []
    count = 0
    jizhunxing = random.randint(0, len(S_Matrix) - 1)
    while len(star_id) <= 10 and count < 10:
        count += 1
        random_starlist = random_selectstar(len(S_Matrix))
        star_l = two2one_dict(dict,  # 添加金字塔容错机制
                              jizhunxing,
                              random_starlist[0],
                              random_starlist[1],
                              random_starlist[2],
                              random_starlist[3],
                              random_starlist[4])
        for ite in star_l:
            star_id.append(ite)
    print(star_id)
    star_id_mode = statistics.mode(star_id)
    star_count = 0
    for star_count_i in range(len(star_id)):
        if star_id_mode == star_id[star_count_i]:
            star_count += 1
    print("置信度为：" + str(star_count / len(star_id)))
    return jizhunxing, star_id_mode
def two2one_dict(dict,jizhun,s1,s2,s3,s4,s5):
    len_dict = len(dict)
    star_l = []
    for i in range(len_dict):
        '''
        二分查找
        '''
        if (S_Matrix[jizhun, s1] in dict[i].values() and
                S_Matrix[jizhun, s2] in dict[i].values() and
                S_Matrix[jizhun, s3] in dict[i].values() and
                S_Matrix[jizhun, s4] in dict[i].values() and
                S_Matrix[jizhun, s5] in dict[i].values()):
            star_l.append(i)
    return star_l

# 只找第一颗星的，以此作为基准星然后扩展其他星
def tri_Ang_dict(dict,star_id_mode, s1, s2, s3):
    star_l = []
    len_dict = len(dict)
    for i in range(len_dict):
        if S_Matrix[s1, s2] in dict[i].values() and S_Matrix[s1, s3] in dict[i].values():
            star1 = dict_binary_search(dict[i], S_Matrix[s1, s2])
            # star1 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s2])]
            # if S_Matrix[s1, s3] in dict[i].values():
            star2 = dict_binary_search(dict[i], S_Matrix[s1, s3])
                # star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s3])]
            if int(star2) in dict[star1]:
                if dict[star1][star2] == S_Matrix[s2, s3]:
                    star_l.append(i)

    return star_l


def tri_Ang(A_M,star_id_mode, s1, s2, s3):
    star_l = []
    # 在附近的星为同一天区
    if star_id_mode >= len(A_M):
        for i in range(len(A_M)):
            for j in range(1, len(A_M[i]), 2):
                if abs(S_Matrix[s1, s2] - A_M[i][j]) < threshold:
                    # print(j)
                    for k in range(1, len(A_M[i]), 2):
                        # print(S_Matrix[0,len(S_Matrix)-1])
                        if abs(S_Matrix[s1, s3] - A_M[i][k]) < threshold:
                            # print(k)
                            kkk = search_index(int(A_M[i][j - 1]), int(A_M[i][k - 1]), A_M)
                            # print(int(A_M[i][j-1]))
                            # print(int(A_M[i][k-1]))
                            if kkk != None:
                                if abs(S_Matrix[s2, s3] - A_M[int(A_M[i][j - 1])][kkk]) < threshold:
                                    star_l.append(i)
                    # print(S_Matrix[2,len(S_Matrix)-1])
                    # print(A_M[int(A_M[i][j-1])][search_index(int(A_M[i][j-1]),int(A_M[i][k-1]),A_M)])
    else: # 应该是后端确定天区然后搜索附近星用的算法，当时有问题，就搁置了，从搜索与天区共同优化，如今为哈希搜索
        star_id_mode_hlist = []
        threshold2 = 0.004
        for i in range(0,len(A_M[star_id_mode]),2):
            star_id_mode_hlist.append(int(A_M[star_id_mode][i]))
        for i in range(len(star_id_mode_hlist)):
            star_id_mode = star_id_mode_hlist[i]
            for j in range(1,len(A_M[star_id_mode]),2):
                if abs(S_Matrix[s1,s2] - A_M[star_id_mode][j]) < threshold2:
                    for k in range(1,len(A_M[star_id_mode]),2):
                        if abs(S_Matrix[s1,s3] - A_M[star_id_mode][k]) < threshold2:
                            kkk = search_index(int(A_M[star_id_mode][j - 1]), int(A_M[star_id_mode][k - 1]), A_M)
                            if kkk != None:
                                if abs(S_Matrix[s2, s3] - A_M[int(A_M[star_id_mode][j - 1])][kkk]) < threshold2:
                                    star_l.append(star_id_mode)
    return star_l

def dict_binary_search(dicti, Sij):
    '''
    二分查找dicti中与S_Matrix[si,sj]相同的value值并返回Key值
    '''
    left = 0
    right = len(dicti)
    keys = list(dicti.keys())
    values = list(dicti.values())
    ans = None
    while left < right:
        middle = (left + right) // 2
        num = values[middle]
        if num < Sij:
            left = middle + 1
        elif num > Sij:
            right = middle
        else:
            return keys[middle]
            break # 很重要，否则陷入无限循环

def four_Ang_dict(dict,s1,s2,s3,s4,s5):
    len_dict = len(dict)
    for i in range(len_dict):
        flag = 0
        '''
        二分查找
        '''
        # if S_Matrix[s1, s2] in dict[i].values():
        #     star1 = dict_binary_search(dict[i], S_Matrix[s1, s2])
        #     if S_Matrix[s1, s3] in dict[i].values():
        #         star2 = dict_binary_search(dict[i], S_Matrix[s1, s3])
        #         if int(star2) in dict[star1]:
        #             if dict[star1][star2] == S_Matrix[s2, s3]:
        #                 flag += 1
        #     elif S_Matrix[s1, s5] in dict[i].values():
        #         star2 = dict_binary_search(dict[i], S_Matrix[s1, s5])
        #         if int(star2) in dict[star1]:
        #             if dict[star1][star2] == S_Matrix[s2, s5]:
        #                 flag += 1
        # if S_Matrix[s1, s3] in dict[i].values():
        #     star1 = dict_binary_search(dict[i], S_Matrix[s1, s3])
        #     if S_Matrix[s1, s4] in dict[i].values():
        #         star2 = dict_binary_search(dict[i], S_Matrix[s1, s4])
        #         if int(star2) in dict[star1]:
        #             if dict[star1][star2] == S_Matrix[s3, s4]:
        #                 flag += 1
        #     if flag == 3:
        #         break
        #     elif S_Matrix[s1, s5] in dict[i].values():
        #         star2 = dict_binary_search(dict[i], S_Matrix[s1, s5])
        #         if int(star2) in dict[star1]:
        #             if dict[star1][star2] == S_Matrix[s3, s5]:
        #                 flag += 1
        #     if flag == 3:
        #         break
        # if S_Matrix[s1, s2] in dict[i].values():
        #     star1 = dict_binary_search(dict[i], S_Matrix[s1, s2])
        #     if S_Matrix[s1, s4] in dict[i].values():
        #         star2 = dict_binary_search(dict[i], S_Matrix[s1, s4])
        #         if int(star2) in dict[star1]:
        #             if dict[star1][star2] == S_Matrix[s2, s4]:
        #                 flag += 1
        if S_Matrix[s1, s2] in dict[i].values():
            star1 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s2])]
            if S_Matrix[s1, s3] in dict[i].values():
                star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s3])]
                if int(star2) in dict[star1]:
                    if dict[star1][star2] == S_Matrix[s2, s3]:
                        flag += 1
            elif S_Matrix[s1, s5] in dict[i].values():
                star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s5])]
                if int(star2) in dict[star1]:
                    if dict[star1][star2] == S_Matrix[s2, s5]:
                        flag += 1
        if S_Matrix[s1, s3] in dict[i].values():
            star1 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s3])]
            if S_Matrix[s1, s4] in dict[i].values():
                star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s4])]
                if int(star2) in dict[star1]:
                    if dict[star1][star2] == S_Matrix[s3, s4]:
                        flag += 1
            if flag == 3:
                break
            elif S_Matrix[s1, s5] in dict[i].values():
                star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s5])]
                if int(star2) in dict[star1]:
                    if dict[star1][star2] == S_Matrix[s3, s5]:
                        flag += 1
            if flag == 3:
                break
        if S_Matrix[s1, s2] in dict[i].values():
            star1 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s2])]
            if S_Matrix[s1, s4] in dict[i].values():
                star2 = list(dict[i].keys())[list(dict[i].values()).index(S_Matrix[s1, s4])]
                if int(star2) in dict[star1]:
                    if dict[star1][star2] == S_Matrix[s2, s4]:
                        flag += 1

        if flag >= 3:
            print("基准星：" + str(i))
            # print("_________________________", time.time() - start)
            return i

# 取四颗星，第一颗为基准星与其他三颗，构成三个三角形，取点分别为：0，int(len（S_Matrix）/3),int(len（S_Matrix）/3*2),len(S_Matrix)。
def four_Ang(A_M, s1, s2, s3, s4):
    len_A_M = len(A_M)
    # for value in dict.values():
    #     flag = 0
    #     if abs(S_Matrix[s1, s2] in value.values:

    for i in range(len_A_M):
        flag = 0
        for j in range(1, len(A_M[i]), 2):
            if abs(S_Matrix[s1, s2] - A_M[i][j]) < threshold:
                for k in range(1, len(A_M[i]), 2):
                    if abs(S_Matrix[s1, s3] - A_M[i][k]) < threshold:
                        kkk = search_index(int(A_M[i][j - 1]), int(A_M[i][k - 1]), A_M)
                        if kkk != None:
                            if abs(S_Matrix[s2, s3] - A_M[int(A_M[i][j - 1])][kkk]) < threshold:
                                flag += 1
            if abs(S_Matrix[s1, s3] - A_M[i][j]) < threshold:
                for k in range(1, len(A_M[i]), 2):
                    if abs(S_Matrix[s1, s4] - A_M[i][k]) < threshold:
                        kkk = search_index(int(A_M[i][j - 1]), int(A_M[i][k - 1]), A_M)
                        if kkk != None:
                            if abs(S_Matrix[s3, s4] - A_M[int(A_M[i][j - 1])][kkk]) < threshold:
                                flag += 1
            if abs(S_Matrix[s1, s2] - A_M[i][j]) < threshold:
                for k in range(1, len(A_M[i]), 2):
                    if abs(S_Matrix[s1, s4] - A_M[i][k]) < threshold:
                        kkk = search_index(int(A_M[i][j - 1]), int(A_M[i][k - 1]), A_M)
                        if kkk != None:
                            if abs(S_Matrix[s2, s4] - A_M[int(A_M[i][j - 1])][kkk]) < threshold:
                                flag += 1
        if flag == 3:
            print("基准星：" + str(i))
            return i


# 根据已知第一颗基准星，搜索其他星
def All_Starid(A_M, jizhunxing, star_id_mode):
    star_a = []
    # 取四颗星，第一颗为基准星与其他三颗，构成三个三角形，取点分别为：0，int(len（S_Matrix）/3),int(len（S_Matrix）/3*2),len(S_Matrix)。
    for s2 in range(1, int((len(S_Matrix) + 1) / 2)):
        s3 = s2 + int(len(S_Matrix) / 2) - 1
        # print(s2)
        for j in range(1, len(A_M[star_id_mode]), 2):
            if abs(S_Matrix[jizhunxing, s2] - A_M[star_id_mode][j]) < threshold:
                # print(j)
                for k in range(1, len(A_M[star_id_mode]), 2):
                    # print(S_Matrix[0,len(S_Matrix)-1])
                    if abs(S_Matrix[jizhunxing, s3] - A_M[star_id_mode][k]) < threshold:
                        # print(k)
                        kkk = search_index(int(A_M[star_id_mode][j - 1]), int(A_M[star_id_mode][k - 1]), A_M)
                        if kkk != None:
                            if abs(S_Matrix[s2, s3] - A_M[int(A_M[star_id_mode][j - 1])][kkk]) < threshold:
                                star_a.append([s2, int(A_M[star_id_mode][j - 1])])
                                star_a.append([s3, int(A_M[star_id_mode][k - 1])])
                            # else:
                            # print("不满足三角关系"+str(s2))
                            # star_a.append([s2, tri_Ang(s2,s3,0)])
                            # star_a.append([s3, tri_Ang(s3,s2,0)])
                        # else:
                        # print("搜寻不到"+str(s2))
                        # star_a.append([s2, tri_Ang(s2, s3, 0)])
                        # star_a.append([s3, tri_Ang(s3, s2, 0)])
            # print("第一步pass")
    return star_a

# 星图帧间可以按照星图2为新星图，做其角距矩阵，然后匹配星图1的，找出来与之前相同的星点，计算其旋转矩阵
def If_Pattern(min_list1, FP_camaxis1, path2):
    # 上一帧与这一帧星图星点矩阵
    threshold = 0.005
    # front_Star_Matrix,min_list1,FP_camaxis1 = Sift_ADMatrix_min(path1)
    now_Frame_Matrix, min_list2, FP_camaxis2 = Sift_ADMatrix_min(path2)
    # print(min_list1)
    # print(min_list2)
    parrten_init = []
    for i in range(len(min_list1)):
        for j in range(len(min_list2)):
            if abs(min_list1[i, 0] - min_list2[j, 0]) < threshold and abs(
                    min_list1[i, 1] - min_list2[j, 1]) < threshold and abs(
                    min_list1[i, 2] - min_list2[j, 2]) < threshold:
                parrten_init.append([i, j])
    # print(parrten_init)
    front_star = []
    now_star = []
    for k in range(len(parrten_init)):
        front_star.append(parrten_init[k][0])
        front_star.append(int(min_list1[parrten_init[k][0]][3]))
        front_star.append(int(min_list1[parrten_init[k][0]][4]))
        front_star.append(int(min_list1[parrten_init[k][0]][5]))
        now_star.append(parrten_init[k][1])
        now_star.append(int(min_list2[parrten_init[k][1]][3]))
        now_star.append(int(min_list2[parrten_init[k][1]][4]))
        now_star.append(int(min_list2[parrten_init[k][1]][5]))
    # print(front_star)
    # print(now_star)
    if len(front_star)!=0:
        front_star_axis = []
        now_star_axis = []
        for m in range(len(front_star)):
            front_star_axis.append(FP_camaxis1[front_star[m]])
            now_star_axis.append(FP_camaxis2[now_star[m]])
        w = 0.01 * np.ones((len(front_star), 3), float)
        if_q = quest(np.array(front_star_axis).T, np.array(now_star_axis).T, w.T)
        if_e = quaternion2euler(if_q)
        return if_q, min_list2, FP_camaxis2
    else:
        if_q, min_list2, FP_camaxis2 = None,None,None
        return if_q, min_list2, FP_camaxis2

def pattern_SEM(ellipse, data, star_file, A_M, dict, img_path, num,region_flag, star_id_mode):
    star_camera_List, star_Cordinate_List, min_list, FP_camaxis,region_flag,star_id_mode = main_star(ellipse,
                                                                            star_file,
                                                                            img_path,
                                                                            A_M,
                                                                            dict,
                                                                            data,
                                                                            num,
                                                                            region_flag,
                                                                            star_id_mode)
    if star_camera_List is None:
        return False,None,None,None,-1
    # quest求解wahba问题
    w = 0.1 * np.ones((len(star_camera_List), 3), float)
    if len(star_camera_List) < 3:
        print("剩余星识别过少！")
        return False,None,None,None,-2

    q_opt = quest(star_camera_List.T, star_Cordinate_List.T, w.T)
    q_opt = np.array(q_opt)
    name_q = list(map(float, img_path.split("_")[7:11]))

    # quest求解后需要与相机与本体的外参旋转矩阵相乘,以此求解本体系相对于J2000的姿态
    rpy = [0,0,0]
    Cam_extM = quaternion2rot(euler2quaternion(rpy))
    q_opt =rot2quaternion(Cam_extM@ quaternion2rot(q_opt))
    # UE左手系转右手系，将Y轴取负后与J2000系相同，然后将命名得出的四元数，qx与qz取负，则xyz旋转矩阵也符合规则
    name_q = np.array(name_q)
    qq_cos = name_q @ q_opt.T / (np.linalg.norm(name_q) * np.linalg.norm(q_opt))
    if qq_cos > 1:
        qq_cos = 2 - qq_cos
    err_i = math.acos(qq_cos)*180/math.pi
    if err_i > 3:
        err_i = 180 - err_i
    print("err_i:" + str(err_i))
    print("真实值四元数:" + str(name_q))
    print("估计四元数为:" + str(q_opt))
    # print("真实值欧拉角:" + str(quaternion2euler(name_q)))
    # print("估计欧拉角为:" + str(quaternion2euler(q_opt)))
    return True,q_opt[0],star_camera_List,region_flag,star_id_mode

if __name__ == '__main__':
   pass