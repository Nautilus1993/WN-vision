# @Author  : Zhao Yutao
# @Time    : 2024/8/1 12:29
# @Function: 各类工具函数
# @mails: zhaoyutao22@mails.ucas.ac.cn
import csv
import math
import os
import re
import shutil
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import spiceypy as spice

def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    # 将字符串按照数字和非数字部分分割，返回分割后的子串列表
    sub_strings = re.split(r'(\d+)', s)
    # 如果当前子串由数字组成，则将它转换为整数；否则返回原始子串
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    # 根据分割后的子串列表以及上述函数的返回值，创建一个新的列表
    # 按照数字部分从小到大排序，然后按照非数字部分的字典序排序
    return sub_strings

# 矩阵转四元数
def rot2quaternion(rot):
    r = R.from_matrix(rot)
    qua = r.as_quat()
    return qua
#四元数转矩阵
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot
#欧拉角转矩阵
def euler2rot(euler):
    r = R.from_euler('zyx', euler, degrees=True)
    rot = r.as_matrix()
    return rot

#四元数转欧拉角 小写外旋大写字母内旋
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('zyx', degrees=True)
    return euler

# 欧拉角转四元数
def euler2quaternion(euler):
    r = R.from_euler('zyx', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def load_furnsh(path):
    bsp_name = os.listdir(path)
    for bsp in bsp_name:
        spice.furnsh(os.path.join(path,bsp))
spice.tkvrsn("TOOLKIT")
load_furnsh('kernel')
def Obv2Tar(obvPositionfilePath,tarPositionfilePath,abfilePath):
    '''
    输入观测星，目标星星历文件计算每一时刻下的测角信息
    :param obvPositionfilePath:星历文件 ****.e
    :param tarPositionfilePath:星历文件 ****.e
    :return: 保存关于测角alpha, belta的文件，abfile
    '''
    obvPositionfile = open(obvPositionfilePath,'r')
    tarPositionfile = open(tarPositionfilePath,'r')
    abfile = open(abfilePath,'w')

    line1 = obvPositionfile.readline()
    line2 = tarPositionfile.readline()
    counts = 1
    # 分别获取观测星、目标星星历文件中时间戳及位置信息
    while line1:
        if counts >= 12 and counts <= 2412:
            # 计算测角信息并保存至abfile中
            timestamp = line1.split(" ")[0]
            obvP = np.array(list(map(float, line1.split(" ")[1:4])))
            tarP = np.array(list(map(float, line2.split(" ")[1:4])))
            delta = tarP - obvP
            alpha = math.atan2(delta[1],delta[0])
            belta = math.atan2(delta[2],math.sqrt(delta[0]**2+delta[1]**2))
            # 保存格式：时间戳 alpha belta
            abfile.write(timestamp + " " + str(alpha) + " " + str(belta) +"\n")
            # obvP_M.append(list(map(float,line1.split(" ")[0:4])))
            # tarP_M.append(list(map(float,line2.split(" ")[0:4])))
        line1 = obvPositionfile.readline()
        line2 = tarPositionfile.readline()
        counts += 1
    abfile.close()

def onlyangle2csv(filepath, csvfilepath):
    '''
    转换成智勋师兄仅测角使用文件
    :param filepath: 原星历.e文件
    :param csvfilepath:输出文件地址
    '''
    efile = open(filepath,'r')
    csvfile = open(csvfilepath,'w',newline='')
    csv_writer = csv.writer(csvfile)
    line = efile.readline()
    count = 1
    while line:
        if count > 11 and count < 2413:
            lines = list(map(float,line.split(" ")[1:7]))
            csv_writer.writerow([count - 12,lines[0],lines[1],lines[2],lines[3],lines[4],lines[5]])
        line = efile.readline()
        count += 1

def obv2Tar1angle(obv1path, obv2path, tarpath, savefile):
    '''
    计算仅观测星1————目标星————观测星2角度计算批处理
    :param obv1path: 观测星1星历文件
    :param obv2path: 观测星2星历文件
    :param tarpath: 目标星星历文件
    :param savefile:  仅测角数据存储文件
    '''
    obv1_xyz = open(obv1path,'r')
    obv2_xyz = open(obv2path,'r')
    tar_xyz = open(tarpath, 'r')
    sava_angle = open(savefile,'w')
    line1 = obv1_xyz.readline()
    line2 = obv2_xyz.readline()
    tar_line = tar_xyz.readline()
    count = 1
    min_angle = 999
    while line1:
        o1_xyz = np.array(list(map(float,line1.split(',')[1:4])))
        o2_xyz = np.array(list(map(float,line2.split(',')[1:4])))
        t_xyz = np.array(list(map(float,tar_line.split(',')[1:4])))
        r1 = o1_xyz - t_xyz
        r2 = o2_xyz - t_xyz
        # 分别计算两个向量的模
        r1_l = np.linalg.norm(r1)
        r2_l = np.linalg.norm(r2)
        # 计算两个向量的点积
        dian = r1.dot(r2)
        # 计算夹角cos值
        cos_ = dian/(r1_l * r2_l)
        # 求得夹角（弧度需要转角度）
        angle = np.arccos(cos_) * 180 / np.pi
        if angle < min_angle :
            min_angle = angle
        sava_angle.write(str(angle) + "\n")
        line1 = obv1_xyz.readline()
        line2 = obv2_xyz.readline()
        tar_line = tar_xyz.readline()
    sava_angle.write(str(min_angle)+"\n")

def csv2e(csvfilepath,efilepath):
    # xyzv = open(csvfilepath, 'r')
    xyzv = csv.reader(open(csvfilepath))
    efile = open(efilepath, 'w')
    efile.write("stk.v.11.6\n"+"BEGIN Ephemeris\n"+"NumberOfEphemerisPoints  3001\n"+"ScenarioEpoch  1 Jul 2024 12:00:00.0000\n"
                    +"InterpolationMethod     Lagrange\n"+"InterpolationOrder      5\n"+"DistanceUnit 			Meters\n"+
                    "CentralBody             Earth\n"+"CoordinateSystem        J2000\n"+"TimeFormat 			Epsec\n"+"EphemerisTimePosVel\n" )
    for oxyz in xyzv:
        oxyz = np.array(list(map(float,oxyz)))
        # print(line)
        # oxyz = np.array(list(map(float, line.split(','))))
        # print(oxyz)
        efile.write(str(oxyz[0])+" "+
                    str(oxyz[1])+" "+
                    str(oxyz[2])+" "+
                    str(oxyz[3])+" "+
                    str(oxyz[4])+" "+
                    str(oxyz[5])+" "+
                    str(oxyz[6])+" "+"\n")
    efile.close()

def symb_Jac_Fun():
    '''
    对观测矩阵求偏导
    '''
    import sympy as sp
    from sympy import symbols, Function, diff
    x, y, z, X, Y, Z = symbols('Px Py Pz Xm Ym Zm')
    f1 = Function('f')(x, y,X,Y)
    f1 = sp.atan2(Y-y,X-x)
    df1_dx = diff(f1, x)
    df1_dy = diff(f1, y)
    # -----------------------***********-----------------
    f2 = Function('f')(x, y, z, X, Y, Z)
    f2 = sp.atan2(Z-z,sp.sqrt((X-x)**2+(Y-y)**2+(Z-z)**2)*sp.sqrt((X-x)**2+(Y-y)**2))
    df2_dx = diff(f2, x)
    df2_dy = diff(f2, y)
    df2_dz = diff(f2, z)
    # -----------------------***********-----------------
    print(df1_dx)
    print(df1_dy)
    print(df2_dx)
    print(df2_dy)
    print(df2_dz)

def sym_dfdxyz_F():
    import sympy as sp
    from sympy import symbols, Function, diff
    x, y, z, Xm, Ym, Zm, Xs,Ys,Zs,Ue,Um,Us = symbols('x y z Xm Ym Zm Xs Ys Zs mu_earth mu_moon mu_sun')
    f1 = Function('f')(x, y, z, Xm, Ym, Zm, Xs,Ys,Zs,Ue,Um,Us)
    f1 = (-Ue*x/sp.sqrt(x**2+y**2+z**2)**3 -
          Um*((x-Xm)/sp.sqrt((x-Xm)**2+(y-Ym)**2+(z-Zm)**2)**3+Xm/sp.sqrt(Xm**2+Ym**2+Zm**2)**3)
          - Us*((x-Xs)/sp.sqrt((x-Xs)**2+(y-Ys)**2+(z-Zs)**2)**3+Xs/sp.sqrt(Xs**2+Ys**2+Zs**2)**3))
    df1_dx = diff(f1,x)
    print(df1_dx)
# sym_dfdxyz_F()

def epoch_10_20_30WKm():
    '''
    求解距地心10、20、30、35w公里的历元，每个历元向后观测10min
    :param TLIEpochfilepath: 地月转移轨道星历文件
    '''
    c_Vec = 299792.458 # 光速 Km/s
    step = 40000
    utc = ['Jan 10, 2026', 'Jan 13, 2026']
    # print(spice.et2utc("759412869.1845883"))
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    # get  times
    epoch_123 = [None,None,None,None] # 存储距地心10、20、30、35w公里的历元
    times = [x * (etTwo - etOne) / step + etOne for x in range(step)]
    for time_ in times:
        positions, lightTimes = spice.spkpos('10001', time_, 'J2000', 'NONE', 'Earth')
        moon_pos, lightTimes2 = spice.spkpos('MOON', time_, 'J2000', 'NONE', 'Earth')
        if 100010 > lightTimes * c_Vec > 99990 and epoch_123[0] == None:
            epoch_123[0] = spice.et2utc(time_, 'C', 3)
            r1 = -positions
            r2 = moon_pos-positions
            # 分别计算两个向量的模
            r1_l = np.linalg.norm(r1)
            r2_l = np.linalg.norm(r2)
            # 计算两个向量的点积
            dian = r1.dot(r2)
            # 计算夹角cos值
            cos_ = dian / (r1_l * r2_l)
            # 求得夹角（弧度需要转角度）
            angle = np.arccos(cos_) * 180 / np.pi
            print("10Wkm同时观测地月满足夹角：",round(angle,3))
        if 200010 > lightTimes * c_Vec > 199990 and epoch_123[1] == None:
            epoch_123[1] = spice.et2utc(time_, 'C', 3)
            r1 = -positions
            r2 = moon_pos - positions
            # 分别计算两个向量的模
            r1_l = np.linalg.norm(r1)
            r2_l = np.linalg.norm(r2)
            # 计算两个向量的点积
            dian = r1.dot(r2)
            # 计算夹角cos值
            cos_ = dian / (r1_l * r2_l)
            # 求得夹角（弧度需要转角度）
            angle = np.arccos(cos_) * 180 / np.pi
            print("20Wkm同时观测地月满足夹角：", round(angle,3))
        if 300010 > lightTimes * c_Vec > 299990 and epoch_123[2] == None:
            epoch_123[2] = spice.et2utc(time_, 'C', 3)
            r1 = -positions
            r2 = moon_pos - positions
            # 分别计算两个向量的模
            r1_l = np.linalg.norm(r1)
            r2_l = np.linalg.norm(r2)
            # 计算两个向量的点积
            dian = r1.dot(r2)
            # 计算夹角cos值
            cos_ = dian / (r1_l * r2_l)
            # 求得夹角（弧度需要转角度）
            angle = np.arccos(cos_) * 180 / np.pi
            print("30Wkm同时观测地月满足夹角：", round(angle,3))
        if 350010 > lightTimes * c_Vec > 349990 and epoch_123[3] == None:
            epoch_123[3] = spice.et2utc(time_, 'C', 3)
            r1 = -positions
            r2 = moon_pos - positions
            # 分别计算两个向量的模
            r1_l = np.linalg.norm(r1)
            r2_l = np.linalg.norm(r2)
            # 计算两个向量的点积
            dian = r1.dot(r2)
            # 计算夹角cos值
            cos_ = dian / (r1_l * r2_l)
            # 求得夹角（弧度需要转角度）
            angle = np.arccos(cos_) * 180 / np.pi
            print("35Wkm同时观测地月满足夹角：", round(angle,3))
    return epoch_123
    # check first few times:
    # times = spice.str2et("2024-04-07T14:37:18.870Z")
    # Run spkpos as a vectorized function
    # positions, lightTimes = spice.spkpos('10001', times, 'J2000', 'NONE', 'Earth')
    # Positions is a 3XN vector of XYZ positions
    # print("Positions: {}".format(positions[0]))
    # Light times is a N vector of time
    # print("Light times:{}".format(lightTimes[0]))
    # clear up the kernels
    # spice.kclear()
    #——————————————画出J2000系下3D图————————————————————
    # positions = positions.T  # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(positions[0], positions[1], positions[2])
    # plt.title("TLI Example from Jan 10, 2026 to Jan 13, 2026")
    # plt.show()

def cal_angle_cameraAB(oc_filepath):
    '''
    求解各工况下，相机AB的夹角、欧拉角并写入Excel文件中
    '''
    all_path = os.listdir(oc_filepath)
    oc_path = []
    for x in all_path:
        if os.path.isdir(os.path.join(oc_filepath,x)):
            oc_path.append(x)
    oc_path = sorted(oc_path,key=natural_sort_key)
    rpy_m = []
    cos_m = []
    for x in oc_path:
        name = os.listdir(os.path.join(oc_filepath,x))
        q1 = np.array(list(map(float,name[0].split("_")[7:11])))
        q2 = np.array(list(map(float,name[1].split("_")[7:11])))
        # rpy1 = quaternion2euler(q1)
        # rpy2 = quaternion2euler(q2)
        # rpy1[2] = rpy1[2] + 23.439291
        # rpy2[2] = rpy2[2] + 23.439291
        # q1 = euler2quaternion(rpy1)
        # q2 = euler2quaternion(rpy2)
        # 求解q1与q2之间的欧拉角转换与矩阵转化、余弦夹角
        # 余弦夹角
        cos_rad = np.dot(q1 , q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
        print(cos_rad)
        # if cos_rad > 0:
        #     cos_rad = 2 - cos_rad
        cos_deg = np.arccos(cos_rad)*180/math.pi
        matrix1 = quaternion2rot(q1) # A相对J2000的旋转矩阵
        matrix2 = quaternion2rot(q2) # B相对J2000的旋转矩阵
        # 计算matrix_J20002B,即B相对J2000的转置（旋转矩阵的逆等于其转置）
        matrix_J20002B =  matrix2.T
        # 求解A相对于B的旋转矩阵
        matrix_AB = np.dot(matrix1,matrix_J20002B)
        # print(matrix_AB)
        rpy_AB = quaternion2euler(rot2quaternion(matrix_AB))
        cos_m.append(round(cos_deg,3))
        rpy_AB = np.around(rpy_AB, decimals=3)
        rpy_m.append(rpy_AB)
        # print(x,"相机AB在此历元下同时观测地月的欧拉角为:",rpy_AB)
        # print(x,"相机AB在此历元下同时观测地月的夹角为:",cos_deg,"°")
        # 创建数据
    print(rpy_m)
    data = {
            '工况': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            '工况形容': ['相机AB同时观测地月'] * 16,  # 假设所有工况的形容都相同
            '欧拉角': rpy_m,
            '夹角 (°)': cos_m
        }

        # 注意：由于'欧拉角'列包含列表，我们需要先处理它以适应DataFrame
        # 这里为了简单起见，我们假设每个工况的欧拉角都是单独的观测，而不是列表中的多个值
        # 但在实际情况下，你可能需要调整数据结构以适应你的需求
        # 假设我们只取每个欧拉角列表的第一个元素（这只是一个简化的例子）
    euler_angles_flattened = [euler for euler in data['欧拉角']]
        # 更新数据以使用展平的欧拉角
    data['欧拉角（°）'] = euler_angles_flattened  # 添加一个新列以包含展平的欧拉角（这里只取第一个元素作为示例）
        # 创建DataFrame（注意：这里我们移除了原始的'欧拉角'列，因为它包含列表）
    df = pd.DataFrame(data, columns=['工况', '工况形容', '夹角 (°)', '欧拉角（°）'])
        # 导出到Excel文件
    df.to_excel('camera_observations.xlsx', index=False)

def quaA2quaB(path1,path2):
    '''
    求解两个四元数之间的旋转关系
    '''
    qA = open(path1,'r')
    qB = open(path2,'r')
    lineA = qA.readline()
    lineB = qB.readline()
    for i in range(25):
        print(lineA)
        q1 = np.array(list(map(float, lineA.split(" ")[4:8])))
        q2 = np.array(list(map(float, lineB.split(" ")[4:8])))
        # rpy1 = quaternion2euler(q1)
        rpy2 = quaternion2euler(q2)
        # rpy1[2] = rpy1[2] + 23.439291
        rpy2[2] = rpy2[2] + 23.439291
        # q1 = euler2quaternion(rpy1)
        q2 = euler2quaternion(rpy2)
        # 求解q1与q2之间的欧拉角转换与矩阵转化、余弦夹角
        # 余弦夹角
        cos_rad = q1 @ q2.T / (np.linalg.norm(q1) * np.linalg.norm(q2))
        # if cos_rad > 0:
        #     cos_rad = 2 - cos_rad
        cos_deg = 2 * math.acos(cos_rad) * 180 / math.pi
        matrix1 = quaternion2rot(q1)  # A相对J2000的旋转矩阵
        matrix2 = quaternion2rot(q2)  # B相对J2000的旋转矩阵
        # 计算matrix_J20002B,即B相对J2000的转置（旋转矩阵的逆等于其转置）
        matrix_J20002B = matrix2.T
        # 求解A相对于B的旋转矩阵
        matrix_AB = np.dot(matrix1, matrix_J20002B)
        # print(matrix_AB)
        rpy_AB = quaternion2euler(rot2quaternion(matrix_AB))
        rpy_AB = np.around(rpy_AB, decimals=3)
        print(rpy_AB)
        lineA = qA.readline()
        lineB = qB.readline()

# 位速矢量转换为六根数
def comp_oe(R, V):
    miu = 3.986e5 #km2/s2
    X = [1, 0, 0]  # y轴方向向量
    Y = [0, 1, 0]  # y轴方向向量
    Z = [0, 0, 1]  # z轴方向向量
    r = np.linalg.norm(R)  # 位置标量
    H = np.cross(R, V)  # 角动量
    h = np.linalg.norm(H)  # 角动量的模
    N = np.cross(Z, H)  # 升交线矢量
    # print('N H',N,H)
    n = np.linalg.norm(N)  # 升交线矢量的模
    # 半长轴 a
    tmp = 2 / r - np.dot(V, V) / miu
    if tmp == 0:  # 抛物线
        a = np.dot(H, H) / miu
    else:
        a = abs(1 / tmp)
    # 离心率 e
    E = ((np.dot(V, V) - miu / r) * R - np.dot(R, V) * V) / miu  # 离心率矢量
    e = np.linalg.norm(E)  # 离心率标量
    if e < 1e-4: e = 0
    # 轨道倾角i
    i = math.acos(np.dot(Z, H) / h)
    # 近心点辐角 w
    # print('e ',e)
    # print('n ',n)
    # print('i', i)
    if e == 0:  # 圆
        # 赤道圆轨道
        if n == 0:
            w = 0.0
        else:
            w = math.acos(np.dot(N, R) / (n * r))
        if np.dot(Z, R) < 0:
            w = 2 * np.pi - w
    else:
        # print(n * e)
        w = math.acos(np.dot(N, E) / (n * e))
        if np.dot(Z, E) < 0:
            w = 2 * np.pi - w
    # 升交点经度 Omega
    if n == 0:
        Omega = 0.0
    else:
        Omega = np.arccos(np.dot(N, X) / n)
    if np.dot(N, Y) < 0:
        Omega = 2 * np.pi - Omega
    # 真近点角 phi
    if e != 0:  # 非圆形轨道
        # print(e * r)
        phi_tmp = np.dot(E, R) / (e * r)
        eps = 1e-6
        if 1.0 < phi_tmp < 1.0 + eps:
            phi_tmp = 1.0
        elif -1.0 - eps < phi_tmp < -1.0:
            phi_tmp = -1.0
        # print(phi_tmp)
        phi = math.acos(phi_tmp)
        if np.dot(R, V) < 0:
            phi = 2 * np.pi - phi
    else:
        phi = 0
    if np.isnan(w):
        w = 0.00000000000000001
    return a, e, i, w, Omega, phi
def beili_txt(tarsat_filepath, save_filepath, tar_type, mul_id = None):
    '''
    UTC 2024-7-1T12:00:00.000
    et 773107269.1840944
    4个目标星，3个观测星
    目标星2，4在900秒进行机动
    目标星1，3在30秒进行释放
    群目标分别为
    1_1，1_2,1_3,1_4,1_5,1_6
    3_1,3_2,3_3,3_4,3_5,3_6
    定义场景使用观测星3——法向释放——>3_1~~~3_6, 机动星4
    '''
    timestamps = None
    sat_count = None
    tar_id = None
    a = None
    e = None
    i = None
    w = None
    Omega = None
    phi = None
    tarsat_ = open(tarsat_filepath,'r')
    save_ = open(save_filepath,'w')
    line1 = tarsat_.readline()
    counts = 1
    if tar_type == 'HC1': # 1 ，3 都为非异常目标
        # 分别获取观测星、目标星星历文件中时间戳及位置信息
        while line1:
            if counts >= 12 and counts <= 2412:
                curline = line1.split(" ")
                timestamps = 773107269.1840944 + float(curline[0])
                sat_count = 1
                tar_id = mul_id
                a,e,i,w,Omega,phi = comp_oe(np.array(list(map(float,curline[1:4])))/1000.0,
                                            np.array(list(map(float,curline[4:7])))/1000.0) # 需要将m转为km

                if counts >= 12+61:
                    sat_count = 8 # 刚释放的时候很难能观测到机动星1+释放星6+目标星1个
                    # tar_type = 'HD2'
                save_.write(tar_type+" "+
                            str(timestamps)+" "+
                            str(sat_count)+" "+
                            str(tar_id)+" "+
                            str(a)+" "+
                            str(e)+" "+
                            str(i)+" "+
                            str(w)+" "+
                            str(Omega)+" "+
                            str(phi)+"\n")
            counts += 1
            line1 = tarsat_.readline()
    if tar_type == 'HD1': # 2,4 都为机动目标
        # 分别获取观测星、目标星星历文件中时间戳及位置信息
        while line1:
            if counts >= 12 and counts <= 2412:
                curline = line1.split(" ")
                timestamps = 773107269.1840944 + float(curline[0])
                sat_count = 1
                tar_id = mul_id
                tar_type = 'HC1'
                a, e, i, w, Omega, phi = comp_oe(np.array(list(map(float,curline[1:4])))/1000.0,
                                            np.array(list(map(float,curline[4:7])))/1000.0) # 需要将m转为km

                if counts >= 12+61:
                    sat_count = 8
                if counts >= 12 + 1801:
                    tar_type = 'HD1'
                save_.write(tar_type + " " +
                            str(timestamps) + " " +
                            str(sat_count) + " " +
                            str(tar_id) + " " +
                            str(a) + " " +
                            str(e) + " " +
                            str(i) + " " +
                            str(w) + " " +
                            str(Omega) + " " +
                            str(phi) + "\n")


            counts += 1
            line1 = tarsat_.readline()
    if tar_type == 'HD2': # 3_1~~~3_6 都为群目标
        # 分别获取观测星、目标星星历文件中时间戳及位置信息
        while line1:
            if counts >= 12 and counts <= 2412:
                curline = line1.split(" ")
                timestamps = 773107269.1840944 + float(curline[0])
                sat_count = 1
                tar_id = mul_id
                tar_type = 'HC1'
                a, e, i, w, Omega, phi = comp_oe(np.array(list(map(float,curline[1:4])))/1000.0,
                                            np.array(list(map(float,curline[4:7])))/1000.0) # 需要将m转为km

                if counts >= 12 + 61:
                    tar_type = 'HD2'
                    sat_count = 8
                save_.write(tar_type + " " +
                            str(timestamps) + " " +
                            str(sat_count) + " " +
                            str(tar_id) + " " +
                            str(a) + " " +
                            str(e) + " " +
                            str(i) + " " +
                            str(w) + " " +
                            str(Omega) + " " +
                            str(phi) + "\n")

            counts += 1
            line1 = tarsat_.readline()
def beili_run():
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3.txt",
        'HC1',3
    )  # 目标星3
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_4.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_4.txt",
        'HD1',4
    )  # 机动星4
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_1.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_1.txt",
        'HD2',31
    )  # 群3_1
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_2.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_2.txt",
        'HD2',32
    )  # 群3_2
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_3.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_3.txt",
        'HD2',33
    )  # 群3_3
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_4.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_4.txt",
        'HD2',34
    )  # 群3_4
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_5.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_5.txt",
        'HD2',35
    )  # 群3_5
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3_6.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_3_6.txt",
        'HD2',36
    )  # 群3_6
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1.txt",
        'HC1', 1
    )  # 目标星3
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_2.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_2.txt",
        'HD1', 2
    )  # 机动星4
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_1.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_1.txt",
        'HD2', 11
    )  # 群3_1
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_2.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_2.txt",
        'HD2', 12
    )  # 群3_2
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_3.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_3.txt",
        'HD2', 13
    )  # 群3_3
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_4.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_4.txt",
        'HD2', 14
    )  # 群3_4
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_5.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_5.txt",
        'HD2', 15
    )  # 群3_5
    beili_txt(
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1_6.e",
        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\beili\sat_1_6.txt",
        'HD2', 16
    )  # 群3_6
# beili_run()
def xigongda_txt(obvsat_filepath,save_filepath, obv_id, tar_id):
    '''
    西工大所需txt文件
    需要目标卫星
    '''
    timestamp = None
    sat_count = None
    x = None
    y = None
    z = None
    rx = None
    ry = None
    rz = None
    tar2obv_alpha = None
    tar2obv_beta = None
    save_ = open(save_filepath,'w')
    obvsat_img = os.listdir(obvsat_filepath)
    obvsat_img = sorted(obvsat_img, key= natural_sort_key)
    init_utc = obvsat_img[0].split("_")[0:7]
    time_step = (spice.utc2et(init_utc[0]+"-"+
                             init_utc[1]+"-"+
                             init_utc[2]+"T"+
                             init_utc[3]+":"+
                             init_utc[4]+":"+
                             init_utc[5]+"."+
                             init_utc[6]) - 773107269.1840944)

    for img in obvsat_img:
        if img.endswith('png'):
            curline = img.split("_")
            print(curline)
            timestamp = spice.utc2et(curline[0]+"-"+
                                 curline[1]+"-"+
                                 curline[2]+"T"+
                                 curline[3]+":"+
                                 curline[4]+":"+
                                 curline[5]+"."+
                                 curline[6])
                                    # - time_step
            sat_count = 3
            if timestamp >= 773107299.1840944:
                sat_count = 3
            q_ = curline[7:11]
            [rx,ry,rz] = quaternion2euler(q_)
            x = curline[11]
            y = curline[12]
            z = curline[13]
            '''
            目前跟踪目标星，因此测角数据不变且一直为0，若以后严谨一点应该利用目标星与
            观测星姿态、位置求解其测角数据，并且释放星的测角数据亦是如此
            '''
            tar2obv_alpha = 0
            tar2obv_beta = 0
            save_.write(
                str(timestamp)+" "+
                str(sat_count)+" "+
                str(tar_id)+" "+
                str(obv_id)+" "+
                str(x)+" "+
                str(y)+" "+
                str(z)+" "+
                str(rx)+" "+
                str(ry)+" "+
                str(rz)+" "+
                str(tar2obv_alpha)+" "+
                str(tar2obv_beta)+"\n"
            )
    save_.close()

# xigongda_txt(r"C:\Users\zhaoy\Desktop\1701-tar3-0.1hz",
#              r"C:\Users\zhaoy\Desktop\fov5_obv1_tar3.txt",
#              1,3)


def xigongda_run():
    xigongda_txt(r"E:\ue5project\3-sat1\1701\fov\normal",
             r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv1_tar1.txt",
             1,1)
    xigongda_txt(r"E:\ue5project\3-sat1\1702\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv2_tar1.txt",
                 2, 1)
    xigongda_txt(r"E:\ue5project\3-sat1\1703\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv3_tar1.txt",
                 3, 1)
    xigongda_txt(r"E:\ue5project\3-sat2\1701\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv1_tar2.txt",
                 1, 2)
    xigongda_txt(r"E:\ue5project\3-sat2\1702\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv2_tar2.txt",
                 2, 2)
    xigongda_txt(r"E:\ue5project\3-sat2\1703\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv3_tar2.txt",
                 3, 2)
    xigongda_txt(r"E:\ue5project\3-sat3\1701\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv1_tar3.txt",
                 1, 3)
    xigongda_txt(r"E:\ue5project\3-sat3\1702\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv2_tar3.txt",
                 2, 3)
    xigongda_txt(r"E:\ue5project\3-sat3\1703\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv3_tar3.txt",
                 3, 3)
    xigongda_txt(r"E:\ue5project\3-sat4\1701\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv1_tar4.txt",
                 1, 4)
    xigongda_txt(r"E:\ue5project\3-sat4\1702\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv2_tar4.txt",
                 2, 4)
    xigongda_txt(r"E:\ue5project\3-sat4\1703\fov\normal",
                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xigongda\obv3_tar4.txt",
                 3, 4)
# xigongda_run()
def xiaoweixing_10simg_(imgpath, saveimgpath):
    '''
    抽取10s的img进行打包
    '''
    targetSat_path = os.listdir(imgpath)
    obv_path_save = ['obv1','obv2','obv3']
    obv_path = ['1701','1702','1703']
    for tar_path in targetSat_path:
        for i in range(len(obv_path)):
            img_path = os.path.join(imgpath,tar_path,obv_path[i],'fov','normal')
            ima_name = os.listdir(img_path)
            ima_name = sorted(ima_name, key = natural_sort_key)
            new_dir = os.path.join(saveimgpath,tar_path,obv_path_save[i])
            for j in range(len(ima_name)):
                if j % 10 == 0:
                    shutil.copy(os.path.join(img_path,ima_name[j]),os.path.join(new_dir, ima_name[j]))
            print(new_dir)
        print(tar_path)
#
# xiaoweixing_10simg_(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\img",
#                     r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg")

def xx_img(imgpath, saveimgpath):
    all_img_path = os.listdir(imgpath)
    all_img_path = sorted(all_img_path,key=natural_sort_key)
    i = 0
    for img_name in all_img_path:
        if i % 2 == 0:
            shutil.copy(os.path.join(imgpath,img_name),os.path.join(saveimgpath,img_name))
        i += 1

# xx_img(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\img\3-sat3\1701\fov\normal",
#        r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\img\3-sat3\tmp\fov\normal")
def xiaoweixing_txt(img_path, save_path):
    '''
    给小卫星的，从副导头开始
    '''
    savefile = open(save_path,'w')
    img_name = os.listdir(img_path)
    img_name = sorted(img_name, key = natural_sort_key)
    id = int(img_path.split("\\")[-2].split("_")[1])
    try:
        for i in range(len(img_name)):
            timestamp = img_name[i].split("_")[0:7]
            timestamp = spice.utc2et(timestamp[0] + "-" +
                                     timestamp[1] + "-" +
                                     timestamp[2] + "T" +
                                     timestamp[3] + ":" +
                                     timestamp[4] + ":" +
                                     timestamp[5] + "." +
                                     timestamp[6])
            xv = None
            ligt = None
            saveStr = None
            if id == 3:
                xv, ligt = spice.spkezr('1711', timestamp, "J2000", 'NONE', 'Earth')
                saveStr = "HC1" + " " + str(timestamp) + " " + str(1) + " " + str(3) + " " + \
                          img_name[i] + " 0 0 " + str(xv[0]) + " " + str(xv[1]) + " " + str(xv[2]) + " " + \
                          str(xv[3]) + " " + str(xv[4]) + " " + str(xv[5]) + "\n"
            else:
                if i < 900:
                    xv, ligt = spice.spkezr('1719', timestamp, "J2000", 'NONE', 'Earth')
                    saveStr = "HC1" + " " + str(timestamp) + " " + str(1) + " " + str(4) + " " + \
                              img_name[i] + " 0 0 " + str(xv[0]) + " " + str(xv[1]) + " " + str(xv[2]) + " " + \
                              str(xv[3]) + " " + str(xv[4]) + " " + str(xv[5]) + "\n"
                else:
                    xv, ligt = spice.spkezr('1719', timestamp, "J2000", 'NONE', 'Earth')
                    saveStr = "HD1" + " " + str(timestamp) + " " + str(1) + " " + str(4) + " " + \
                              img_name[i] + " 0 0 " + str(xv[0]) + " " + str(xv[1]) + " " + str(xv[2]) + " " + \
                              str(xv[3]) + " " + str(xv[4]) + " " + str(xv[5]) + "\n"
            savefile.write(saveStr)
    except Exception as e:
        print(e)
    savefile.close()

# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_3\obv1",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar3_obv1.txt")
# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_3\obv2",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar3_obv2.txt")
# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_3\obv3",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar3_obv3.txt")
# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_4\obv1",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar4_obv1.txt")
# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_4\obv2",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar4_obv2.txt")
# xiaoweixing_txt(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\10simg\target_4\obv3",
#                 r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\xiaoweixing\tar4_obv3.txt")

if __name__ == '__main__':
    # cal_angle_cameraAB(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\data\DXO-EM")
    # print(epoch_10_20_30WKm())
    # quaA2quaB(r'C:\Users\zhaoy\PycharmProjects\EarthMoon\TestData\es2024-04-18.txt',
    #           r'C:\Users\zhaoy\PycharmProjects\EarthMoon\TestData\gt2024-04-18.txt')
    # symb_Jac_Fun()
    # csv2e(r"C:\Users\zhaoy\Desktop\zhixun\sat1_J2000RV(1).csv",
    #       r"C:\Users\zhaoy\Desktop\zhixun\z1.e")
    # csv2e(r"C:\Users\zhaoy\Desktop\zhixun\sat2_J2000RV(1).csv",
    #       r"C:\Users\zhaoy\Desktop\zhixun\z2.e")
    # csv2e(r"C:\Users\zhaoy\Desktop\zhixun\Tar1_01_J2000RV(1).csv",
    #       r"C:\Users\zhaoy\Desktop\zhixun\zTar.e")
    # Obv2Tar(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\observer_1.e",
    #         r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\satellite_3.e",
    #         "t_a_b.txt")
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\observer_1.e",
    #               'epochanglefileetc\observer_1.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\observer_2.e",
    #               'epochanglefileetc\observer_2.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\observer_3.e",
    #               'epochanglefileetc\observer_3.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_1.e",
    #               'epochanglefileetc\satellite_1.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_2.e",
    #               'epochanglefileetc\satellite_2.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_3.e",
    #               'epochanglefileetc\satellite_3.csv')
    # onlyangle2csv(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\utils\epochanglefileetc\epochfile\satellite_4.e",
    #               'epochanglefileetc\satellite_4.csv')

    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\satellite_1.csv",
    #               r"1.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\satellite_2.csv",
    #               r"2.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\satellite_3.csv",
    #               r"3.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\satellite_4.csv",
    #               r"4.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_1.csv",
    #               r"5.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_2.csv",
    #               r"6.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_3.csv",
    #               r"7.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_1.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_4.csv",
    #               r"8.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_1.csv",
    #               r"9.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_2.csv",
    #               r"10.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_3.csv",
    #               r"11.txt")
    # obv2Tar1angle(r"epochanglefileetc\observer_2.csv",
    #               r"epochanglefileetc\observer_3.csv",
    #               r"epochanglefileetc\satellite_4.csv",
    #               r"12.txt")
    pass