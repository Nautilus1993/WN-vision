# @Author  : Zhao Yutao
# @Time    : 2024/4/1 12:29
# @Function: 测试redis服务端发送图像
# @mails: zhaoyutao22@mails.ucas.ac.cn
import datetime
import math
import os
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import spiceypy as spice
from Net.predict import net_pre
from data.center_calculate import geo_calculate, moon_calculate
from star_ephemeris.Pattern.AngDis_Pattern_des import pattern_SEM
from utils.utils import natural_sort_key, load_furnsh
import setting.settingsPara as para

spice.tkvrsn("TOOLKIT")
load_furnsh('kernel')


# 四元数转矩阵
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot


# 矩阵转四元数
def rot2quaternion(rot):
    r = R.from_matrix(rot)
    qua = r.as_quat()
    return qua


# 四元数转欧拉角 外旋
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('zyx', degrees=True)
    return euler


# 欧拉角转四元数
def euler2quaternion(euler):
    r = R.from_euler('zyx', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion


# 旋转矩阵转欧拉角
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def outest_cnt(contours):
    max_c = 0
    # x = []
    cnt = np.array([[[0, 0]], [[0, 1]], [[0, 2]], [[0, 3]], [[0, 4]]])
    for i in range(len(contours)):
        # x.append(len(contours[i]))
        if max_c <= len(contours[i]):
            max_c = len(contours[i])
            cnt = contours[i]
    # print(cnt)
    return cnt


class ESQuest:
    '''
    添加此四元数估计对象，将每次的四元数估计结果可以写进去，也可以读出来
    为了应对每次如果想要用真值姿态来去计算测角信息时就可以不用每次都进入星图模块进行计算，减少实验时间
    可以用于测角信息计算
    可以用于四元数误差计算
    '''

    def __init__(self):
        self.path = "TestData/esq_record"
        self.logname = (para.testdata.split('/')[-1].split('\\')[-1] + "_" +
                        para.testdata.split('/')[-1].split('\\')[-2] + "_" +
                        para.testdata.split('/')[-1].split('\\')[-3] + "_" + str(para.starmap_pattern) + ".txt")
        self.name = None
        self.q = None

    def write(self, img_name, esq, type=0):
        file = open(os.path.join(self.path, self.logname), 'a')
        # print(file.name)
        file.write(img_name + " " + str(esq[0]) + " " + str(esq[1]) + " " + str(esq[2]) + " " + str(esq[3]) + " " + str(
            type) + "\n")
        file.close()

    def clear(self):
        file = open(os.path.join(self.path, self.logname), 'w')
        file.close()

    def read(self, img_name):
        file = open(os.path.join(self.path, self.logname))
        line = file.readline()
        es_q = None
        while line:
            all = line.split()
            if all[0] == img_name:
                es_q = list(map(float, all[1:5]))
            line = file.readline()
        file.close()
        return es_q

'''
全局
'''
into_star_pattern = None
esq_record = None
measurement_txt = open("TestData/measurement/singlemeasurement" + str(datetime.date.today()) + ".txt", "a")
gttum_txt = open("TestData/gt/singlegt" + str(datetime.date.today()) + ".txt", "a")
estum_txt = open("TestData/es/singlees" + str(datetime.date.today()) + ".txt", "a")
gt_moon_file = open("TestData/gt/gtmoon" + str(datetime.date.today()) + ".txt", "a")
es_moon_file = open("TestData/es/esmoon" + str(datetime.date.today()) + ".txt", "a")

if para.q_bool == False:
    esq_record = ESQuest()
    esq_recordpath = 'TestData/esq_record'
    logname = (para.testdata.split('/')[-1].split('\\')[-1] +
               "_" + para.testdata.split('/')[-1].split('\\')[-2] +
               "_" + para.testdata.split('/')[-1].split('\\')[-3] + "_" + str(para.starmap_pattern) + '.txt')
    # print(os.path.join(esq_recordpath,logname))
    if os.path.exists(os.path.join(esq_recordpath, logname)):
        esq_recordfile = open(os.path.join(esq_recordpath, logname), 'r')
        allline = esq_recordfile.readlines()
        if len(allline) < len(os.listdir(para.testdata)) * 0.9:
            esq_record.clear()
            into_star_pattern = True
        else:
            into_star_pattern = False
        esq_recordfile.close()
    else:
        into_star_pattern = True
else:
    into_star_pattern = False


def core_extraction(data, star_file, A_M, dict, img_path, camera_f, r, region_flag, star_id_mode, ekf_filter, model):
    '''
    紧耦合星图与地月识别
    '''
    img = cv2.imread(img_path) # socket超参直接传到这个图像数组
    # J2000系下Camera相对于Earth真实四元数
    gt_q = np.array(list(map(float, img_path.split("_")[7:11])))  # socket超参直接传到这个四元数姿态
    # rpy = [-150, 65, -160]
    # xxxxx = quaternion2rot(euler2quaternion(rpy))
    # gt_q = rot2quaternion(quaternion2rot(gt_q) @ xxxxx)
    # J2000系真实坐标
    gt_xyz = np.array(list(map(float, img_path.split("_")[11:14]))) # socket超参直接传到这个位置速度
    # print(gt_xyz)
    # 距地心真实距离
    gt_dis = np.linalg.norm(gt_xyz)
    # 高像素 宽像素 通道数
    try:
        rows, cols, ch = img.shape
    except Exception as e:
        print(e)
        return None, None, None, None, None, None, None, None, None
    # 高斯Canny提取边缘
    edge = guass_canny(img)
    # 提取轮廓
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓 第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    dst = np.ones(img.shape, dtype=np.uint8)
    cv2.drawContours(dst, contours, -1, (0, 255, 0), 1)
    # 选取最外层轮廓

    cnt = outest_cnt(contours)
    (x_circle, y_circle), radius_circle = cv2.minEnclosingCircle(cnt)

    a_sque = np.squeeze(cnt)
    # 找出当前多边形的边界框
    min_x = min(point[0] for point in a_sque)
    max_x = max(point[0] for point in a_sque)
    min_y = min(point[1] for point in a_sque)
    max_y = max(point[1] for point in a_sque)
    cx_m = (min_x + max_x) / 2
    cy_m = (min_y + max_y) / 2
    ellipse = cv2.fitEllipse(cnt)
    # ellipse = (0,0),(0,0),0
    print("椭圆参数（x,y,a,b,φ）：" + str(ellipse))
    # 将椭圆Mask方程利用在星图识别中，只提取椭圆外的兴趣点
    es_q_star = None
    flag = False
    # 测试用，如果用真值姿态与伪真值椭圆参数则将下面一行进行注释掉
    if into_star_pattern == True:
        flag, es_q_star, star_camera_List, region_flag, star_id_mode = pattern_SEM(ellipse,
                                                                                   data,
                                                                                   star_file,
                                                                                   A_M,
                                                                                   dict,
                                                                                   img_path,
                                                                                   para.starmap_pattern,
                                                                                   region_flag,
                                                                                   star_id_mode)

        if star_id_mode >= 0:
            es_q = es_q_star
            ekf_filter.es_q = es_q
        elif star_id_mode == -1:
            es_q = gt_q
            ekf_filter.es_q = es_q
            esq_record.write(img_path, es_q, 1)  # 加个表示代表识别是否成功，0为成功，1为失败，默认为成功
            print("星图识别算法匹配失败！！！")
    else:
        if para.q_bool == False:
            es_q = esq_record.read(img_path)
        else:
            es_q = gt_q
        # print("暂用真值")
    if para.withdraw_r_method == 0:
        # 椭圆模型直接法
        return Direct_method_ellipse_modeling(img_path, ellipse, camera_f, cols, rows, dst,
                                              es_q, gt_xyz, gt_dis, r, region_flag, star_id_mode, ekf_filter, cx_m,
                                              cy_m, x_circle, y_circle, radius_circle)
    elif para.withdraw_r_method == 1:
        # 网络法
        return yolo_bbox_center_keypoint(img_path, ellipse, camera_f, cols, rows, dst,
                                              es_q, gt_xyz, gt_dis, r, region_flag, star_id_mode, ekf_filter, model)


def draw_onecore(cols, rows, Xe, Ye, dst):
    ecore_ = (Xe / para.Pixel_w + cols / 2, rows / 2 - Ye / para.Pixel_h), (10, 10), 0
    cv2.ellipse(dst, ecore_, (255, 255, 255), -1, 2)
    dst = cv2.resize(dst, (1200, 1200))
    cv2.imshow("Newton", dst)
    cv2.imwrite(r"data\Earth_onecore.png", dst)
    cv2.waitKey()


def draw_twocore(Xc, Yc, cols, rows, Xe, Ye, dst):
    '''
    勾勒质心与椭圆中心
    '''
    core_ = (Xc / para.Pixel_w + cols / 2, rows / 2 - Yc / para.Pixel_h), (10, 10), 0
    ecore_ = (Xe / para.Pixel_w + cols / 2, rows / 2 - Ye / para.Pixel_h), (10, 10), 0
    cv2.ellipse(dst, core_, (0, 255, 0), -1, 2)
    cv2.ellipse(dst, ecore_, (255, 255, 255), -1, 2)
    dst = cv2.resize(dst, (1200, 1200))
    cv2.imshow("Ellipse", dst)
    cv2.imwrite(r"data\Earth_twocore.png", dst)
    cv2.waitKey()


def yolo_bbox_center_keypoint(img_path, ellipse, camera_f, cols, rows, dst,
                              es_q, gt_xyz, gt_dis, r, region_flag, star_id_mode, ekf_filter, model):
    filename = img_path.split("/")[-1].split("\\")[-1].split("_")
    utc = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + filename[
        4] + ":" + filename[5] + "." + filename[6]
    print(utc)
    time1 = spice.utc2et(utc)
    id, bbox_center_net, ellipse_center_net, obj_core_net, rad_ = net_pre(model, img_path, resize=para.resize)
    if bbox_center_net is None:
        return None, None, None, None, None, None, es_q, region_flag, star_id_mode
    else:
        Xe, Ye = (bbox_center_net[0] - para.Width_p / 2) * para.Pixel_w, (
                    para.Heigh_p / 2 - bbox_center_net[1]) * para.Pixel_h
    # print(Xe,Ye)
    # time.sleep(3)
    '''
    提取观测量时并未发生跳变，
    为何在EKF中途会出现跳变，
    这是个非常大的疑惑？？？而且之后还能再拉回来！明日探究！    
    2024.11.28
    因为位置初值的选择与滤波协方差矩阵的参数不太正确，而且，从测距估计量来看，对于Ed并不友好，Net提取的非常平稳，很大原因从测距这个飘出去的
    '''
    # 将椭圆中心像素坐标，长短轴，与图像坐标系X轴的旋转φ角度，赋值下来
    (Xc, Yc), (ellipse_a, ellipse_b), angle_R = ellipse
    Reds = [camera_f / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            -Xe / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            Ye / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2)]
    es_dis = r / math.sin(rad_)
    gt_RM1 = quaternion2rot(es_q)
    T = - gt_RM1 @ (np.array(Reds) * es_dis)
    T_m2c = -T
    a_theta = math.atan2(T_m2c[1], T_m2c[0])
    b_theta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
    if r > 4000:
        residual = es_dis - gt_dis
        cos_a = gt_xyz[0] * T[0] + gt_xyz[1] * T[1] + gt_xyz[2] * T[2]
        cos_b = np.linalg.norm(gt_xyz) * np.linalg.norm(T)
        cos = cos_a / cos_b
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        Pointing_precision = abs(math.acos(cos) * 180 / math.pi)
        print("像素角精度为:{:.4f}°".format(Pointing_precision))
        ekf_filter.core_x = Xe / para.Pixel_w
        ekf_filter.core_y = Ye / para.Pixel_h
        # print(T)
        ekf_filter.ekf_update_single(timestamps=time1,
                                     singlex=T[0],
                                     singley=T[1],
                                     singlez=T[2],
                                     position_x_measure=T[0] * 1000,
                                     position_y_measure=T[1] * 1000,
                                     position_z_measure=T[2] * 1000,
                                     e_rho=es_dis * 1000,
                                     e_alpha=a_theta,
                                     e_beta=b_theta,
                                     rho_err=5000000, m2c_err=0.001, P_xyz_cov=3000000)

    else:
        # 当是地球图像则是距J2000系的，当是月球图像，还需要继续转一下得出月球此时的历元
        positions2Earth, lightTimes = spice.spkpos('MOON', time1, 'J2000', 'NONE', 'Earth')
        positions2Moon, lightTimes = spice.spkpos(para.sat_id_str, time1, 'J2000', 'NONE', 'Moon')
        # 不能直接相加，还需要考虑此时月球与地球的姿态旋转情况，确定完月球质心指向后，求取月球J2000系下卫星坐标，然后再相加
        T_EarthJ2000 = positions2Earth + T
        residual = np.linalg.norm(T_EarthJ2000) - gt_dis
        gt_xyz_moon = positions2Moon
        cos_a = gt_xyz_moon[0] * T[0] + gt_xyz_moon[1] * T[1] + gt_xyz_moon[2] * T[2]
        cos_b = np.linalg.norm(gt_xyz_moon) * np.linalg.norm(T)
        cos = cos_a / cos_b
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        Pointing_precision = abs(math.acos(cos) * 180 / math.pi)
        print("像素角精度为:{:.4f}°".format(Pointing_precision))
        T = np.array(T_EarthJ2000)
        ekf_filter.core_x = Xe / para.Pixel_w
        ekf_filter.core_y = Ye / para.Pixel_h
        ekf_filter.ekf_update_single(timestamps=time1,
                                     singlex=T[0],
                                     singley=T[1],
                                     singlez=T[2],
                                     position_x_measure=T[0] * 1000,
                                     position_y_measure=T[1] * 1000,
                                     position_z_measure=T[2] * 1000,
                                     m_rho=es_dis * 1000,
                                     m_alpha=a_theta,
                                     m_beta=b_theta,
                                     rho_err=2000000, m2c_err=0.001, P_xyz_cov=1400000)
    if para.draw_center == True:
        draw_twocore(Xc, Yc, cols, rows, Xe, Ye, dst)  # 画地心与椭圆中心
    print("J2000系下真实坐标(km)：" + str(gt_xyz))
    print("EKF_result(xyzv): " +
          str(ekf_filter.position_x_posterior_est) + " " +
          str(ekf_filter.position_y_posterior_est) + " " +
          str(ekf_filter.position_z_posterior_est) + " " +
          str(ekf_filter.speed_x_posterior_est) + " " +
          str(ekf_filter.speed_y_posterior_est) + " " +
          str(ekf_filter.speed_z_posterior_est))
    return T, es_dis, a_theta, b_theta, Pointing_precision, residual, es_q, region_flag, star_id_mode


def Direct_method_ellipse_modeling(img_path,
                                   ellipse,
                                   camera_f,
                                   cols,
                                   rows,
                                   dst,
                                   es_q,
                                   gt_xyz,
                                   gt_dis,
                                   r,
                                   region_flag,
                                   star_id_mode,
                                   ekf_filter, cx_m, cy_m, x_circle, y_circle, radius_circle):
    '''
    椭圆模型直接法
    '''
    filename = img_path.split("/")[-1].split("\\")[-1].split("_")
    utc = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + filename[
        4] + ":" + filename[5] + "." + filename[6]
    time1 = spice.utc2et(utc)
    # 将椭圆中心像素坐标，长短轴，与图像坐标系X轴的旋转φ角度，赋值下来
    (Xc, Yc), (ellipse_a, ellipse_b), angle_R = ellipse
    # angle_R=0
    xc_e, yc_e = Xc, Yc
    # print("图像坐标系坐标",Xc,Yc)
    angle_R = angle_R * math.pi / 180.0
    # print("椭圆参数（x,y,a,b,φ）：" + str(ellipse))
    cv2.ellipse(dst, ellipse, (0, 0, 255), -1, 2)
    '''
    建立像平面参考坐标系 OsXsrYsr：原点与像平面坐标系重合，x 轴正向沿长轴向右，y 轴正向沿短轴向上，
    则像平面参考坐标系相当于由像平面坐标系逆时针旋转φ角度后得到，两个坐标系间的转换矩阵可表示为：
    '''
    TOs_O = np.array([[math.cos(angle_R), math.sin(angle_R)], [-math.sin(angle_R), math.cos(angle_R)]])
    # print(TOs_O)
    # 像平面参考坐标系下 OsXsrYsr，椭圆中心坐标为将像素坐标系转为标准2048的图像坐标系，转为
    Xc, Yc = (Xc - cols / 2) * para.Pixel_w, (rows / 2 - Yc) * para.Pixel_h
    # print("像平面坐标系坐标",Xc,Yc)
    ellipse_a, ellipse_b = ellipse_a * para.Pixel_w / 2, ellipse_b * para.Pixel_w / 2  # 求的半长轴！！！不是全长轴！！！a为短轴，b为长轴
    [Xm, Ym] = TOs_O @ np.array([Xc, Yc])
    # print("像平面参考坐标系坐标", Xm, Ym)

    norm_cfym = math.sqrt(camera_f ** 2 + Ym ** 2)
    # norm_cfym = camera_f
    '''
    假设光学系统焦距为 f，由图4中几何关系可知，地球临边视线角（半角）ρ 及视场轴线与地心物像连线之间夹角ηe分别可表示为：
    '''
    rou = (math.atan2(ellipse_b + Xm, norm_cfym) + math.atan2(ellipse_b - Xm, norm_cfym)) / 2
    ne = (math.atan2(ellipse_b + Xm, norm_cfym) - math.atan2(ellipse_b - Xm, norm_cfym)) / 2

    # 由于地心坐标始终位于椭圆的长轴上，即yoer=ym，因此，在像平面参考坐标系下地心坐标可表示为：
    Xoer, Yoer = norm_cfym * math.tan(ne), Ym
    # print(Xoer)
    # Xoer = camera_f*math.tan(ne)
    # print("像平面参考坐标系坐标", Xoer, Yoer)

    # 将像平面参考坐标系下的地心坐标（xm，ym）转换到像平面坐标系，转换矩阵为：
    TOs_O_ = np.linalg.inv(TOs_O)
    # print(TOs_O_)

    # 像平面坐标系下的地心坐标（xe，ye）可表示为：
    [Xe, Ye] = TOs_O_ @ np.array([Xoer, Yoer]).T
    # Xe,Ye = Xc,Yc
    '''
    姿态与质心提取采用真值
    '''
    if para.core_draw_cal == True:
        uv, uv_p = moon_calculate(img_path)
        if r > 4000:
            uv, uv_p = geo_calculate(img_path)
        (Xe, Ye) = uv
        (xc_c, yc_c) = uv_p
        '''
        数形结合变换的方式对椭圆参数是否精确，对质心是否精准，目标是为了提取质心和通过比大小计算距离
        如果数形结合BBOX、椭圆变换、圆拟合与重投影质心计算误差不大于N像素即可
        然后通过这类思想来设计利用几何约束达到我们的目标网络
        '''
        print("椭圆中心：x:" + str(xc_e - 1024) + "_y:" + str(1024 - yc_e))
        print("重投影计算质心：x:" + str(xc_c) + "_y:" + str(yc_c))
        print("圆拟合中心：x:" + str(x_circle - 1024) + "_y:" + str(1024 - y_circle))
        print("BBox中心：x:" + str(cx_m - 1024) + "_y:" + str(1024 - cy_m))
        print('提取质心像素差影响（椭圆拟合有无thate）---------------' + str(
            np.linalg.norm(np.array((xc_c - xc_e + 1024, yc_c - 1024 + yc_e)))))
        print('提取质心像素差影响（利用BBox确定）---------------' + str(
            np.linalg.norm(np.array((xc_c - cx_m + 1024, yc_c - 1024 + cy_m)))))
        print('提取质心像素差影响（利用圆拟合）' + str(
            np.linalg.norm(np.array((xc_c - x_circle + 1024, yc_c - 1024 + y_circle)))))
    # print("像平面坐标系坐标", Xe, Ye)
    '''
    由于光学系统投影中心坐标在 OsXsYsZs 坐标系下可表示为（0，0，f），则可推导出在坐标系OsXsYsZs下的单位地心矢量（red）s：
    Reds = [-Xe/math.sqrt(Xe**2+Ye**2+camera_f**2),-Ye/math.sqrt(Xe**2+Ye**2+camera_f**2),camera_f/math.sqrt(Xe**2+Ye**2+camera_f**2)]
    还需要转成与J2000系同样指向的
    '''
    Reds = [camera_f / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            -Xe / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            Ye / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2)]
    es_dis = r / math.sin(rou)
    es_dis += para.dis_compensation
    '''
    距离采用真值
    '''
    if para.es_dis2gt_dis == True:
        es_dis = gt_dis
        if r < 4000:
            positions2Moon, lightTimes = spice.spkpos(para.sat_id_str, time1, 'J2000', 'NONE', 'Moon')
            es_dis = np.linalg.norm(positions2Moon)
    # print("单位矢量",Reds)
    '''
      Camera2J2000的旋转矩阵R1已知，平移T1未知    J20002Camera的旋转矩阵R2为R1的逆，平移T2为地心单位矢量乘距离
      T1就等于位姿矩阵的逆的最后一列
    '''
    gt_RM1 = quaternion2rot(es_q)
    T = - gt_RM1 @ (np.array(Reds) * es_dis)

    # 视线矢量应该为
    T_m2c = -T
    a_theta = math.atan2(T_m2c[1], T_m2c[0])
    b_theta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
    residual = None

    if r > 4000:
        print("单位地心矢量（red）s" + str(Reds))
        print("卫星距地心真实距离(km)：" + str(gt_dis))
        print("卫星距地心估计距离（km）：" + str(es_dis))
        residual = es_dis - gt_dis
        print("估计残差(km)：" + str(residual))
        print("J2000系下真实坐标(km)：" + str(gt_xyz))
        print("J2000系下估计坐标（km）" + str(T))
        cos_a = gt_xyz[0] * T[0] + gt_xyz[1] * T[1] + gt_xyz[2] * T[2]
        cos_b = np.linalg.norm(gt_xyz) * np.linalg.norm(T)
        cos = cos_a / cos_b
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        Pointing_precision = abs(math.acos(cos) * 180 / math.pi)
        print('分辨率为：{:.4f}m@10km'.format(math.acos(cos) * gt_dis / (gt_dis / 10.0) * 1000.0))
        print("像素角精度为:{:.4f}°".format(Pointing_precision))
        ekf_filter.core_x = Xe / para.Pixel_w
        ekf_filter.core_y = Ye / para.Pixel_h
        ekf_filter.ekf_update_single(timestamps=time1,
                                     singlex=T[0],
                                     singley=T[1],
                                     singlez=T[2],
                                     position_x_measure=T[0] * 1000,
                                     position_y_measure=T[1] * 1000,
                                     position_z_measure=T[2] * 1000,
                                     e_rho=es_dis * 1000,
                                     e_alpha=a_theta,
                                     e_beta=b_theta)
    else:
        # 当是地球图像则是距J2000系的，当是月球图像，还需要继续转一下得出月球此时的历元
        positions2Earth, lightTimes = spice.spkpos('MOON', time1, 'J2000', 'NONE', 'Earth')
        positions2Moon, lightTimes = spice.spkpos(para.sat_id_str, time1, 'J2000', 'NONE', 'Moon')
        # 不能直接相加，还需要考虑此时月球与地球的姿态旋转情况，确定完月球质心指向后，求取月球J2000系下卫星坐标，然后再相加
        T_EarthJ2000 = positions2Earth + T
        residual = np.linalg.norm(T_EarthJ2000) - gt_dis
        print("单位月心矢量（red）s:" + str(Reds))
        print("卫星距地心真实距离(km)：" + str(gt_dis))
        print("卫星距地心估计距离（km）：" + str(np.linalg.norm(T_EarthJ2000)))
        print("距地估计残差(km)：" + str(residual))
        print("J2000系下真实坐标(km)：" + str(gt_xyz))
        print("J2000系下估计坐标（km）" + str(T_EarthJ2000))
        print("卫星距月心真实距离（km）：" + str(np.linalg.norm(positions2Moon)))
        print("卫星距月心估计距离（km）：" + str(es_dis))
        print("距月估计残差(km)：" + str(es_dis - np.linalg.norm(positions2Moon)))
        print("月球J2000系下真实坐标（km）" + str(positions2Moon))
        print("月球J2000系下估计坐标（km）" + str(T))
        gt_moon_file.write(str(time1) + " " +
                           str(positions2Moon[0]) + " " +
                           str(positions2Moon[1]) + " " +
                           str(positions2Moon[2]) + " " +
                           filename[7] + " " +
                           filename[8] + " " +
                           filename[9] + " " +
                           filename[10] + "\n")
        es_moon_file.write(str(time1) + " " +
                           str(T[0]) + " " +
                           str(T[1]) + " " +
                           str(T[2]) + " " +
                           filename[7] + " " +
                           filename[8] + " " +
                           filename[9] + " " +
                           filename[10] + "\n")

        gt_xyz = positions2Moon
        cos_a = gt_xyz[0] * T[0] + gt_xyz[1] * T[1] + gt_xyz[2] * T[2]
        cos_b = np.linalg.norm(gt_xyz) * np.linalg.norm(T)
        cos = cos_a / cos_b
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        # T = T_EarthJ2000
        # es_dis = np.linalg.norm(T_EarthJ2000)
        Pointing_precision = abs(math.acos(cos) * 180 / math.pi)
        T = np.array(T_EarthJ2000)
        print('分辨率为：{:.4f}m@10km'.format(math.acos(cos) * gt_dis / (gt_dis / 10.0) * 1000.0))
        print("像素角精度为:{:.4f}°".format(Pointing_precision))
        ekf_filter.core_x = Xe / para.Pixel_w
        ekf_filter.core_y = Ye / para.Pixel_h
        ekf_filter.ekf_update_single(timestamps=time1,
                                     singlex=T[0],
                                     singley=T[1],
                                     singlez=T[2],
                                     position_x_measure=T[0] * 1000,
                                     position_y_measure=T[1] * 1000,
                                     position_z_measure=T[2] * 1000,
                                     m_rho=es_dis * 1000,
                                     m_alpha=a_theta,
                                     m_beta=b_theta)
    if para.draw_center == True:
        draw_twocore(Xc, Yc, cols, rows, Xe, Ye, dst)  # 画地心与椭圆中心
    print("EKF_result(xyzv): " +
          str(ekf_filter.position_x_posterior_est) + " " +
          str(ekf_filter.position_y_posterior_est) + " " +
          str(ekf_filter.position_z_posterior_est) + " " +
          str(ekf_filter.speed_x_posterior_est) + " " +
          str(ekf_filter.speed_y_posterior_est) + " " +
          str(ekf_filter.speed_z_posterior_est))
    return T, es_dis, a_theta, b_theta, Pointing_precision, residual, es_q, region_flag, star_id_mode

def guass_canny(img):
    '''
    Canny边缘提取+高斯滤波
    通过调整Kisze，sigmax参数，以及Canny阈值1与阈值2的数值大小
    初步结论，近一点的设置Ksize大一点，sigmax需要进行微调，阈值1和2大致在200~255之间
    这样就可以利用机器学习的思想进行学习最优参数
    '''
    try:
        gaussian_blur = cv2.GaussianBlur(img, (15, 15), 0.)
        edge = cv2.Canny(gaussian_blur, 0, 250)
        # gaussian_blur = cv2.GaussianBlur(img, (11, 11), 3) # 救命数据集sigma为3
        # edge = cv2.Canny(gaussian_blur, 200, 250)

    except Exception as e:
        print(e)
        edge = None
    return edge


def More_Core_extra(data, star_file, A_M, dict, model, img_path, camera_f, Earth_r, Moon_r, ekf_filter):
    global measurement_txt, gttum_txt, estum_txt
    region_flag = False
    star_id_mode = 0
    T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q = None, None, None, None, None, None, None
    start_time = time.time()
    filename = []
    timestames_list = []
    if img_path.endswith(".png"):
        cls = MoonOrEarth(img_path)
        if cls == 0:
            T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q, region_flag, star_id_mode = core_extraction(
                data,
                star_file,
                A_M,
                dict,
                img_path,
                camera_f,
                Earth_r,
                region_flag,
                star_id_mode,
                ekf_filter,
                model)
        elif cls == 1:
            T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q, region_flag, star_id_mode = core_extraction(
                data,
                star_file,
                A_M,
                dict,
                img_path,
                camera_f,
                Moon_r,
                region_flag,
                star_id_mode,
                ekf_filter,
                model)
        elif cls == 2:
            pass
        else:
            print("Error")
        filename = img_path.split('\\')[-1].split('_')
        # print(filename)
        utc = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + \
              filename[4] + ":" + filename[5] + "." + filename[6]
        timestamps = spice.str2et(utc)
        timestames_list.append(timestamps)
        qx = filename[7].split(" ")[0]
        qy = filename[8].split(" ")[0]
        qz = filename[9].split(" ")[0]
        qw = filename[10].split(" ")[0]
        x = filename[11]
        y = filename[12]
        z = filename[13]
        # csv_writer.writerow([timestamps, x, y, z, qx, qy, qz, qw])
        gttum_txt.write(
            str(timestamps) +
            " " + x + " " + y + " " + z + " " +
            qx + " " + qy + " " + qz + " " + qw + "\n")
        estum_txt.write(str(timestamps) + " " +
                        str(T[0]) + " " +
                        str(T[1]) + " " +
                        str(T[2]) + " " +
                        str(es_q[0]) + " " +
                        str(es_q[1]) + " " +
                        str(es_q[2]) + " " +
                        str(es_q[3]) + "\n")
        measurement_txt.write(str(timestamps) + " " +  # 时间s
                              str(es_dis) + " " +  # 距离km
                              str(a_theta) + " " +  # 弧度rad
                              str(b_theta) + " " +  # 弧度rad
                              " " + x + " " + y + " " + z + " " +  # km
                              str(Pointing_Precision) + " " +
                              str(residual) + " " +
                              str(T[0]) + " " +
                              str(T[1]) + " " +
                              str(T[2]) + " " + "\n")
        print("单帧花费（s）：" + str(time.time() - start_time))
        return T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q
    else:
        img_name = os.listdir(img_path)
        img_name = sorted(img_name, key=natural_sort_key)
        print(len(img_name))
        xyz = []
        es_dis_list = []
        a_theta_list = []
        b_theta_list = []
        Pointing_Precision_list = []
        residual_list = []
        es_q_list = []
        for i in range(len(img_name)):
            print(i)
            start_time = time.time()
            cls = MoonOrEarth(os.path.join(img_path, img_name[i]))
            # 1为地球，-1为月球，0为都含，其他则是都不含
            if cls == 0:
                T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q, region_flag, star_id_mode = core_extraction(
                    data,
                    star_file,
                    A_M,
                    dict,
                    os.path.join(img_path, img_name[i]),
                    camera_f,
                    Earth_r,
                    region_flag,
                    star_id_mode,
                    ekf_filter, model)
                if T is None:
                    continue
                xyz.append(T)
                es_dis_list.append(es_dis)
                a_theta_list.append(a_theta)  # 方位角
                b_theta_list.append(b_theta)  # 俯仰角
                Pointing_Precision_list.append(Pointing_Precision)  # 指向精度
                residual_list.append(residual)  # 残差
                es_q_list.append(es_q)  # 四元数估计值
            elif cls == 1:
                T, es_dis, a_theta, b_theta, Pointing_Precision, residual, es_q, region_flag, star_id_mode = core_extraction(
                    data,
                    star_file,
                    A_M,
                    dict,
                    os.path.join(img_path, img_name[i]),
                    camera_f,
                    Moon_r,
                    region_flag,
                    star_id_mode,
                    ekf_filter, model)
                if T is None:
                    continue
                xyz.append(T)
                es_dis_list.append(es_dis)
                a_theta_list.append(a_theta)  # 方位角
                b_theta_list.append(b_theta)  # 俯仰角
                Pointing_Precision_list.append(Pointing_Precision)  # 指向精度
                residual_list.append(residual)  # 残差
                es_q_list.append(es_q)  # 四元数估计值
            elif cls == 2:
                pass
            else:
                print("Error")
            print("单帧花费（s）：" + str(time.time() - start_time))

        measurement_txt = open("TestData/measurement/measurement" + str(datetime.date.today()) + ".txt", "w+")
        gttum_txt = open("TestData/gt/gt" + str(datetime.date.today()) + ".txt", "w+")
        estum_txt = open("TestData/es/es" + str(datetime.date.today()) + ".txt", "w+")

        for i in range(len(xyz)):
            filename.append(img_name[i].split('_'))
        for i in range(len(xyz)):
            utc = filename[i][0] + "-" + filename[i][1] + "-" + filename[i][2] + "T" + filename[i][3] + ":" + \
                  filename[i][4] + ":" + filename[i][5] + "." + filename[i][6]
            timestamps = spice.str2et(utc)
            timestames_list.append(timestamps)
            qx = filename[i][7].split(" ")[0]
            qy = filename[i][8].split(" ")[0]
            qz = filename[i][9].split(" ")[0]
            qw = filename[i][10].split(" ")[0]
            x = filename[i][11]
            y = filename[i][12]
            z = filename[i][13]
            # csv_writer.writerow([timestamps, x, y, z, qx, qy, qz, qw])
            gttum_txt.write(
                str(timestamps) +
                " " + x + " " + y + " " + z + " " +
                qx + " " + qy + " " + qz + " " + qw + "\n")
            estum_txt.write(str(timestamps) + " " +
                            str(xyz[i][0]) + " " +
                            str(xyz[i][1]) + " " +
                            str(xyz[i][2]) + " " +
                            str(es_q_list[i][0]) + " " +
                            str(es_q_list[i][1]) + " " +
                            str(es_q_list[i][2]) + " " +
                            str(es_q_list[i][3]) + "\n")
            if i == 0:
                measurement_txt.write(str(timestamps) + " " +
                                      str(es_dis_list[i]) + " " +
                                      str(a_theta_list[i]) + " " +
                                      str(b_theta_list[i]) + " " +
                                      "0" + " " + x + " " + y + " " + z + " " + "0 0 0 "
                                      + str(Pointing_Precision_list[i]) + " " +
                                      str(residual_list[i]) + " " +
                                      str(xyz[i][0]) + " " +
                                      str(xyz[i][1]) + " " +
                                      str(xyz[i][2]) + " " + "0 0 0" + "\n")
            else:
                delta_t = timestamps - timestames_list[i - 1]
                measurement_txt.write(str(timestamps) + " " +  # 时间s
                                      str(es_dis_list[i]) + " " +  # 距离km
                                      str(a_theta_list[i]) + " " +  # 弧度rad
                                      str(b_theta_list[i]) + " " +  # 弧度rad
                                      str((es_dis_list[i] - es_dis_list[i - 1]) / delta_t)  # 径向速度km/s
                                      + " " + x + " " + y + " " + z + " " +  # km
                                      str((float(x) - float((filename[i - 1][11]))) / delta_t) + " " +  # km/s
                                      str((float(y) - float((filename[i - 1][12]))) / delta_t) + " " +  # km/s
                                      str((float(z) - float((filename[i - 1][13]))) / delta_t) + " " +
                                      str(Pointing_Precision_list[i]) + " " +
                                      str(residual_list[i]) + " " +
                                      str(xyz[i][0]) + " " + str(xyz[i][1]) + " " + str(xyz[i][2]) + " " +
                                      str((xyz[i][0] - xyz[i - 1][0] / delta_t)) + " " +
                                      str((xyz[i][1] - xyz[i - 1][1] / delta_t)) + " " +
                                      str((xyz[i][2] - xyz[i - 1][2] / delta_t)) + "\n")
        gttum_txt.close()
        estum_txt.close()
        measurement_txt.close()
        print("over")


# 给一张图像，判断图像中地球，月球，还是都不含，还是都含
def MoonOrEarth(img_path):
    flag = 0
    if 'moon' in img_path.split('/')[-1].split('\\')[-1] or 'Moon' in img_path.split('/')[-1].split('\\')[-1] or 'MOON' in img_path.split('/')[-1].split('\\')[-1]:
        flag = 1
    return flag

if __name__ == '__main__':
    pass