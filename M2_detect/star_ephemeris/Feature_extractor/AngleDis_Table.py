import pandas as pd
import numpy as np
import os
import csv
import math

#求解每颗星一定视场内的角距星进行保存，否则则不存储。利用三维数组（a,b,c）
def Ang_Dis_List(fov_2):
    fov = np.linalg.norm(fov_2)
    txtname = "AngDis"+str(fov_2[0])+"_"+str(fov_2[1])+"_L_index_sorted.txt"
    data = np.loadtxt(open(r"../Star_Datasets/hyg_dro.csv", "rb"), delimiter=",", skiprows=1, usecols=[2, 3, 4])  # 获取所有星点J2000下xyz数据
    Star_num = len(data)
    # print(Star_num)
    with open(txtname, 'w') as f:
        for i in range(Star_num):
            A_D_L_i = []
            A_D_L_ad = []
            for j in range(Star_num):
                cos_a = data[i, 0] * data[j, 0] + data[i, 1] * data[j, 1] + data[i, 2] * data[j, 2]
                cos_b = math.sqrt(data[i, 0] ** 2 + data[i, 1] ** 2 + data[i, 2] ** 2) * math.sqrt(
                    data[j, 0] ** 2 + data[j, 1] ** 2 + data[j, 2] ** 2)
                cos = cos_a / cos_b
                if cos > 1:
                    cos = 1
                elif cos < -1:
                    cos = -1
                # print(abs(math.acos(cos)*180/math.pi))
                ad = abs(math.acos(cos) * 180 / math.pi)
                if ad<fov:
                    A_D_L_i.append(j)
                    A_D_L_ad.append(ad)
            zipped = zip(A_D_L_i, A_D_L_ad)
            sort_A = sorted(zipped, key=lambda x: (x[1], x[0]))
            # print(sort_A)
            result = zip(*sort_A)
            # print(result)
            i_list ,ad_list = [list(x) for x in result]
            # print(sort_A)
            # print(A_D_L)
            for i in range(len(i_list)):
                f.write(str(i_list[i])+" "+str(ad_list[i])+" ")
            f.write("\n")

        f.close()
    print("角距库已写入文件！")


if __name__ == '__main__':
    # Angle_Dis_Matrix()
    width_cam = 17.64 # mm
    height_cam = 13.32 # mm
    f_cam = 50 # mm
    fov_w = 2*math.atan2(width_cam/2,f_cam)*180/math.pi
    fov_h = 2*math.atan2(height_cam/2,f_cam)*180/math.pi
    Ang_Dis_List([3,3])
    # Angle_Dis_Matrix_FOV(20*math.sqrt(2))