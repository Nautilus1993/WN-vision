import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
#ue为左手系，x前y右z上，相机坐标系为左手z前y上x右

#矩阵转四元数
def rot2quaternion(rot):
    r = R.from_matrix(rot)
    qua = r.as_quat()
    return qua
#四元数转矩阵
def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot
#四元数转欧拉角
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('zyx', degrees=True)
    return euler
#欧拉角转四元数
def euler2quaternion(euler):
    r = R.from_euler('zyx', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

#旋转矩阵转欧拉角
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#旋转向量转旋转矩阵
def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm

#旋转矩阵转旋转向量
def rot2rotvector(rot):
    Rv = cv2.Rodrigues(rot)[0]
    return Rv

# rotvector = np.array([[0.223680285784755, 0.240347886848190, 0.176566110650535]])
# print(rotvector2rot(rotvector))
# print(rot2rotvector(rotvector2rot(rotvector)))

def rot2euler(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = math.atan2(R[1, 0], R[0, 0]) * 180 / np.pi
    else:
        x = math.atan2(-R[1, 2], R[1, 1]) * 180 / np.pi
        y = math.atan2(-R[2, 0], sy) * 180 / np.pi
        z = 0

    return np.array([x, y, z])

def Ang_TwoAxis_Matrix(one_Cordinate_List,two_Cordinate_List):
    #求解两组数据质心
    # print(one_Cordinate_List[0]/len(one_Cordinate_List[0]))
    P1 = [sum(one_Cordinate_List[0])/len(one_Cordinate_List[0]),
          sum(one_Cordinate_List[1])/len(one_Cordinate_List[0]),sum(one_Cordinate_List[2])/len(one_Cordinate_List[0])]
    P2 = [sum(two_Cordinate_List[0])/len(two_Cordinate_List[0]),
          sum(two_Cordinate_List[1])/len(two_Cordinate_List[0]),sum(two_Cordinate_List[2])/len(two_Cordinate_List[0])]
    #将两个数据集的质心移动至同一个点，即只存在于一个旋转的转换关系
    # print(P1)
    # print(P2)
    one_Cordinate_List = one_Cordinate_List.astype(float)
    two_Cordinate_List = two_Cordinate_List.astype(float)
    for i in range(len(one_Cordinate_List)):
        for j in range(len(one_Cordinate_List[i])):
            one_Cordinate_List[i, j] = one_Cordinate_List[i,j] - P1[i]
            # print(one_Cordinate_List[i, j])
    for i in range(len(two_Cordinate_List)):
        for j in range(len(two_Cordinate_List[i])):
            two_Cordinate_List[i, j] = two_Cordinate_List[i, j] - P2[i]
    # print(one_Cordinate_List)
    # print(two_Cordinate_List)
    #通过SVD算法计算旋转和平移关系
    H = np.zeros((3,3))
    # print(H)
    for i in range(len(one_Cordinate_List[0])):
        # print(np.array([list(one_Cordinate_List[:, i])]).T)
        # print(two_Cordinate_List.T[i])
        H += np.array([list(one_Cordinate_List[:, i])]).T*two_Cordinate_List.T[i]
    # print(H)
    U,s,V = np.linalg.svd(H)#SVD分解结果，SVD奇异值越多，则代表信息量越大
    #步骤四：计算旋转和平移关系
    R = V@U.T
    T = -R@P1+P2
    euler_ = rot2euler(R)
    quaternion_ = euler2quaternion(euler_)
    # print(U)
    # print(s)
    # print(V)
    print("旋转矩阵:")
    print(R)
    print("欧拉角：")
    print(euler_)
    print("四元数：")
    print(quaternion_)
    print("平移矩阵")
    print(T)




if __name__ == '__main__':
    # 已知一组点在不同坐标系下的坐标，求解两坐标系的变换关系
    one_Cordinate_List = [[-9.85813364, -7.14581311, 70.9],
                          [-6.89153746, 2.81623229, 70.9],
                          [-2.9877156, 9.34759602, 70.9],
                          [2.14810669, -7.22347498, 70.9],
                          [4.85476702, 1.90729275, 70.9],
                          [5.35392165, -5.6183368, 70.9],
                          [10.00053287, 5.1699236, 70.9],
                          [10.72468907, -0.16453564, 70.9],
                          [11.04902625, 1.57162696, 70.9]]
    star_id = [3060,4552,2966,4419,4390,2995,3984,3349,3146]
    data = np.loadtxt(open(r"../Star_Datasets/hyg_dro.csv", "rb"), delimiter=",", skiprows=1, usecols=[2, 3, 4])  # 获取所有星点J2000下xyz数据
    star_Cordinate_List = []
    for j in range(len(star_id)):
        star_Cordinate_List.append(data[star_id[j]])
    star_Cordinate_List = np.array(star_Cordinate_List)
    one_Cordinate_List = np.array(one_Cordinate_List)
    for i in range(len(one_Cordinate_List)):
        #转换单位向量 ,还需要令相机系的Z与Y交换一下，与UE系相同
        # ue为左手系，x前y右z上，相机坐标系为左手z前y上x右
        one_Cordinate_List[i,0]=one_Cordinate_List[i,0]/math.sqrt(one_Cordinate_List[i,0]**2+one_Cordinate_List[i,1]**2+one_Cordinate_List[i,2]**2)
        one_Cordinate_List[i,1]=one_Cordinate_List[i,2]/math.sqrt(one_Cordinate_List[i,0]**2+one_Cordinate_List[i,1]**2+one_Cordinate_List[i,2]**2)
        one_Cordinate_List[i, 2] = one_Cordinate_List[i, 1] / math.sqrt(one_Cordinate_List[i, 0] ** 2 + one_Cordinate_List[i, 1] ** 2 + one_Cordinate_List[i, 2] ** 2)
    # print(one_Cordinate_List)
    # print(star_Cordinate_List)
    # Ang_TwoAxis_Matrix(star_Cordinate_List.T,one_Cordinate_List.T)
    # #选取点不能共线，最好大于四个点
    # one =[[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
    # two =[[0,-1,0],[1,0,0],[0,0,1],[1,-1,1]]
    # print(np.array(one).T)
    # print(np.array(two).T)
    # Ang_TwoAxis_Matrix(np.array(one).T,np.array(two).T)
    q = [-0.000010623500969,
   0.000043358812283,
  -0.704002231853617,
   0.710197758059177]
    print(quaternion2euler(q))
    rpy = [90, 0, 0]
    euler_q = euler2quaternion(rpy)
    q2Rm = quaternion2rot(euler_q)
    Rm2Rv = rot2rotvector(q2Rm)
    print(Rm2Rv)
