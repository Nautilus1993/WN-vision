import time
import cv2
import numpy as np
import math
import pandas as pd

# cv2.imshow('SIFT', img)
# cv2.waitKey(0)
from matplotlib import pyplot as plt
# img = cv2.imread(r'C:\Users\zhaoy\Desktop\datasets\2024_2_29_11_24_47_698_-0.00878_-0.021112_-0.38388_0.9231_-6998.5374_143.0894_0_.png')
# img1 = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create()
# kp = sift.detect(gray, None)
#
# cv2.drawKeypoints(gray, kp, img)
# cv2.drawKeypoints(gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# kp = cv2.KeyPoint_convert(kp)  #特征点坐标
# print(kp)
# plt.subplot(121), plt.imshow(img),
# plt.title('Dstination'), plt.axis('off')
# plt.subplot(122), plt.imshow(img1),
# plt.title('Dstination'), plt.axis('off')
# plt.show()

img = cv2.imread(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\data\RealTimedata\alashan\pic\2024_8_4_23_6_38_0_0_0_0_0_0_0_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = cv2.GaussianBlur(rgb, (1,1), 0) #可以更改核大小
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(result, None)  # 找到关键点
img = cv2.drawKeypoints(result, kp, img)  # 绘制关键点
cv2.imshow("1",img)
cv2.imwrite('now.jpg',img)
cv2.waitKey()
kp = cv2.KeyPoint_convert(kp)  #特征点坐标
FP_camaxis=[]
count = 0
ellipse =  (0.7691040039062, 0.5841674804688), (0.43287658691406, 0.9629669189453), 0.94558715820312
(Xc, Yc), (ellipse_a, ellipse_b), angle_R = ellipse
for n,i in enumerate(kp):
    if i not in kp[:n]:

        # print(i)
        if (i[0] - Xc) ** 2 + (i[1] - Yc) ** 2 > (ellipse_b/2*1.5) ** 2:
            count += 1
            FP_camaxis.append([(i[0]-1024)*25/2048,(1024-i[1])*25/2048,70.9])
# FP_camaxis = np.zeros((count,3))  # 构造n*n矩阵
FP_camaxis=np.array(FP_camaxis)
print(FP_camaxis)
S_Matrix = np.zeros((len(FP_camaxis),len(FP_camaxis)))
for i in range(len(FP_camaxis)):
    for j in range(i, len(FP_camaxis)):
        cos_a = FP_camaxis[i, 0] * FP_camaxis[j, 0] + FP_camaxis[i, 1] * FP_camaxis[j, 1] + FP_camaxis[i, 2] * FP_camaxis[j, 2]
        cos_b = math.sqrt(FP_camaxis[i, 0] ** 2 + FP_camaxis[i, 1] ** 2 + FP_camaxis[i, 2] ** 2) * math.sqrt(
            FP_camaxis[j, 0] ** 2 + FP_camaxis[j, 1] ** 2 + FP_camaxis[j, 2] ** 2)
        cos = cos_a / cos_b
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        # print(abs(math.acos(cos) * 180/math.pi))
        ad = abs(math.acos(cos) * 180 / math.pi)
        S_Matrix[i, j] = ad
        S_Matrix[j, i] = ad
print(S_Matrix)
print(len(FP_camaxis))