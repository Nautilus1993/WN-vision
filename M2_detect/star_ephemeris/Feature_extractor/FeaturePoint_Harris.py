import cv2
import numpy as np

def harris_corner_detector(path, window_size=2, k=0.04, threshold=0.01):
    # 计算图像梯度
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算协方差矩阵M
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    height, width = image.shape
    offset = window_size // 2
    corner_response = np.zeros((height, width))

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # 计算局部窗口内的协方差矩阵的特征值
            Sxx = np.sum(Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Sxy = np.sum(Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Syy = np.sum(Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1])

            # 计算特征值
            det_M = (Sxx * Syy) - (Sxy ** 2)
            trace_M = Sxx + Syy
            corner_response[y, x] = det_M - k * (trace_M ** 2)

    # 对角点响应进行阈值处理
    corner_response[corner_response < threshold * corner_response.max()] = 0
    corner = []
    for i in range(len(np.where(corner_response>0)[1])):
        corner.append([np.where(corner_response>0)[1][i],np.where(corner_response>0)[0][i]])
    # print(corner)
    return np.array(corner)

# # 读取图像
# image = cv2.imread(r'C:\Users\zhaoy\PycharmProjects\EarthMoon\data\dxoAstar\2025_06_24_9_52_26_000_0.1441_-0.5656_0.1154_0.8037_210707.54396881646_301860.42229888565_172167.49214162916_star_.png', cv2.IMREAD_GRAYSCALE)
#
# # 使用Harris角点检测器


from PIL import Image, ImageDraw
import numpy as np
import os


def draw_points_on_image(image_path, points, point_color=(255, 0, 0),
                         point_size=5, output_path=None):
    """
    在图像上根据坐标点绘制点并保存

    参数:
    image_path: 原始图像路径
    points: 二维数组或列表，包含点的坐标 [x, y]
    point_color: 点的颜色 (R, G, B)
    point_size: 点的直径（像素）
    output_path: 输出图像路径（默认为原始路径添加后缀）
    """
    # 1. 打开原始图像
    with Image.open(image_path) as img:
        # 保留原始分辨率
        orig_width, orig_height = img.size

        # 转换为RGB模式（确保支持颜色）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 2. 创建绘图对象
        draw = ImageDraw.Draw(img)

        # 3. 绘制所有点
        for point in points:
            # 确保坐标是整数且在图像范围内
            x = max(0, min(orig_width - 1, int(point[0])))
            y = max(0, min(orig_height - 1, int(point[1])))

            # 绘制圆形点
            draw.ellipse([
                (x - point_size // 2, y - point_size // 2),  # 左上角
                (x + point_size // 2, y + point_size // 2)  # 右下角
            ], fill=point_color)

        # 4. 保存结果（保持原始分辨率）
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_with_points{ext}"

        img.save(output_path)
        print(f"处理完成! 图像已保存至: {output_path}")
        return output_path


# 示例使用
if __name__ == "__main__":
    point_array = harris_corner_detector(
        r'C:\Users\zhaoy\PycharmProjects\EarthMoon\data\dxoAstar\2025_06_24_9_52_26_000_0.1441_-0.5656_0.1154_0.8037_210707.54396881646_301860.42229888565_172167.49214162916_star_.png')

    # 3. 在图像上绘制点
    draw_points_on_image(
        r'C:\Users\zhaoy\PycharmProjects\EarthMoon\data\dxoAstar\2025_06_24_9_52_26_000_0.1441_-0.5656_0.1154_0.8037_210707.54396881646_301860.42229888565_172167.49214162916_star_.png'
        ,point_array)


# print(corners)
# 显示图像和检测的角点
# import matplotlib.pyplot as plt
# plt.imshow(image, cmap='gray')
# plt.scatter(corners[:,0], corners[:,1], c='r', marker='o')
# plt.show()
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread(r'C:\Users\zhaoy\PycharmProjects\EarthMoon\data\phaseDiff0304-0316\2025_03_04_08_59_20_7086_-0.1871163279315812_-0.032101638985922124_0.2669536173818053_-0.9448241798147353_185851.22_263824.37_135228.18_moon_.png')
# img1 = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create()
# kp = sift.detect(gray, None)
#
#
# cv2.drawKeypoints(gray, kp, img)
# cv2.drawKeypoints(gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# kp = cv2.KeyPoint_convert(kp)  #特征点坐标
# print(kp)
# plt.figure(num=1, figsize=(5, 10))
# plt.subplot(211), plt.imshow(img),
# plt.title('Dstination'), plt.axis('off')
# plt.subplot(212), plt.imshow(img1),
# plt.title('Dstination'), plt.axis('off')
# plt.show()
# K = [[1744.92206139719, 0, 720.0], [0, 1746.58640701753, 540.0], [0, 0, 1]]
# K = np.array(K)
# T = [[-0.241705640],[0.218143689],[4.89465484]]
# T = np.array(T)
# Q = [0.056550817,0.78191275,0.553326166,-0.281504193]
# from scipy.spatial.transform import Rotation
# R = Rotation.from_quat(Q)
# rot = R.as_matrix()
#
# RT = np.append(rot,T,axis=1)
# keni = (K@RT)[:,0:3]
# qici = np.array([(K@RT)[:,3]]).T
# # print(K@RT)
# # print(keni)
# # print(qici)
# # print(kp)
# XYZ1 = np.zeros((len(kp),3))
# for i in range(len(kp)):
#     kp_ = list(kp[i])
#     kp_.append(1)
#     kp_ = np.array([kp_]).T
#     qici = kp_-qici
#     Kni = np.linalg.inv(keni)
#     XYZ1[i] = (Kni@qici).T
# print(kp)
# print(XYZ1)
