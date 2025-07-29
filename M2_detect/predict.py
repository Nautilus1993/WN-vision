# 测试图片
# import csv
import datetime
import math
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import re
from ultralytics import YOLO
import cv2
import numpy as np

camera_f = 70.8910227452213691374302
# 关键点的顺序
keypoint_list = ["core", "ellipse", 'GT']
# 关键点的颜色
keypoint_color = [(0, 5, 205), (0, 255, 0), (255, 0, 0)]
def net_pre(model,media_path,resize):
    # 获取类别
    objs_labels = model.names  # get class labels
    # print(objs_labels)
    # 类别的颜色
    class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0)]
    # 读取图片
    frame = cv2.imread(media_path)
    scale_img = frame.shape[0] / resize
    frame = cv2.resize(frame, (resize, resize))
    bbox_center = None
    ellipse_center = None
    obj_core = None
    rad_ = None
    # rotate
    # 检测
    result = list(model(frame, conf=0.3, stream=True))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
    # print(result)
    boxes = result.boxes  # Boxes object for bbox outputs
    boxes = boxes.cpu().numpy()  # convert to numpy array

    id = None
    # print(len(boxes.data))
    # 遍历每个框
    l,t,r,b =None,None,None,None
    for box in boxes.data:
        l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
        conf, id = box[4:] # confidence, class
        bbox_center = np.array([(l+r)/2*scale_img, (t+b)/2*scale_img])
        id = int(id)
        # 绘制框
        cv2.rectangle(frame, (l,t), (r,b), class_color[id], 2)
        # 绘制类别+置信度（格式：98.1%）
        cv2.putText(frame, f"{objs_labels[id]} {conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color[id], 2)
    # # if id == 0:
    # # 转换颜色空间
    # image = frame[t:b,l:r]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 将图像转换为二维数据
    # pixels = image.reshape(-1, 3)
    # # 使用 KMeans 聚类
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(pixels)
    # # 获取主要颜色
    # colors = kmeans.cluster_centers_
    # colors = colors.astype(int)
    Gc = None
    if l is not None:
        rad_ = math.atan2(((r-l)/2 + (b-t)/2)/2*scale_img*(25/2048),camera_f)
    # 遍历keypoints
    keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = keypoints.cpu().numpy()  # convert to numpy array
    center_x, center_y = None,None
    # draw keypoints, set first keypoint is red, second is blue
    for keypoint in keypoints.data:
        for i in range(len(keypoint)):
            if i == 0:
                x,y,c = keypoint[i]
                obj_core = np.array([x*scale_img, y*scale_img])
                x,y = int(x), int(y)
                # cv2.circle(frame, (x,y), 5, keypoint_color[0], -1)
                # cv2.putText(frame, f"{keypoint_list[0]}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             keypoint_color[0], 2)

            if i == 1:
                x, y, c = keypoint[i]
                ellipse_center = np.array([x * scale_img, y * scale_img])
                x, y = int(x), int(y)
                # cv2.circle(frame, (x, y), 10, keypoint_color[i], -1)
                # cv2.putText(frame, f"{keypoint_list[1]}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, keypoint_color[i], 2)

        if len(keypoint) >= 2:
            # draw arrow line from tail to half between head and tail
            x1,y1,c1 = keypoint[0]
            x2,y2,c2 = keypoint[1]
            center_x, center_y = (x1+x2)/2, (y1+y2)/2
            cv2.arrowedLine(frame, (int(x2),int(y2)), (int(center_x), int(center_y)), (255,0,255), 4, line_type=cv2.LINE_AA, tipLength=0.1)

    if bbox_center is not None:
        '''
        是否有必要将预测椭圆中心、质心、BBOX中心取平均值，可去可加
        '''
        if ellipse_center is None:
            obj_core = (bbox_center+obj_core)/2
        else:
            obj_core = (bbox_center + ellipse_center + obj_core) / 3
        # print(obj_core)
        # pass
        cv2.circle(frame, (round(obj_core[0]/scale_img), round(obj_core[1]/scale_img)), 5, keypoint_color[0], -1)
        cv2.putText(frame, f"{keypoint_list[0]}", (round(obj_core[0]/scale_img), round(obj_core[1]/scale_img)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, keypoint_color[0], 2)
    # if 'snl' in media_path:
    #     cv2.circle(frame, (512, 512), 5, keypoint_color[2], -1)
    #     cv2.putText(frame, f"{keypoint_list[2]}", (512-50,532), cv2.FONT_HERSHEY_SIMPLEX, 2, keypoint_color[2], 2)

    # save image
    # if not os.path.exists(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\Net\result\{}".format(media_path.split('\\')[-2]+str(resize))):
    #     os.mkdir(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\Net\result\{}".format(media_path.split('\\')[-2]+str(resize)))
    #     os.mkdir(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\Net\bbox\{}".format(media_path.split('\\')[-2]+str(resize)))
    # cv2.imwrite(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\Net\result\{}\{}".format(media_path.split('\\')[-2]+str(resize),media_path.split('\\')[-1]), frame)
    # cv2.imwrite(r"C:\Users\zhaoy\PycharmProjects\EarthMoon\Net\bbox\{}\{}".format(media_path.split('\\')[-2]+str(resize),media_path.split('\\')[-1]), frame[t:b,l:r])
    # print("save result.jpg")
    if ellipse_center is None:
        ellipse_center = obj_core
    return id,bbox_center,ellipse_center,obj_core,rad_

def ellipse_directly_cal(media_path,r):
    img = cv2.imread(media_path)
    gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)
    edge = cv2.Canny(gaussian_blur, 200, 250)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.array([[[0,0]],[[0,1]],[[0,2]],[[0,3]],[[0,4]]])
    max_c = 0
    x = []
    for i in range(len(contours)):
        x.append(len(contours[i]))
        if max_c <= len(contours[i]):
            max_c = len(contours[i])
            cnt = contours[i]
    a_sque = np.squeeze(cnt)
    min_x = min(point[0] for point in a_sque)
    max_x = max(point[0] for point in a_sque)
    min_y = min(point[1] for point in a_sque)
    max_y = max(point[1] for point in a_sque)
    bbox_center_ed = np.array([(min_x+max_x)/2,(min_y+max_y)/2])
    ellipse = cv2.fitEllipse(cnt)
    (Xc, Yc), (ellipse_a, ellipse_b), angle_R = ellipse
    ellipse_center = np.array([Xc,Yc])
    xc_e, yc_e = Xc, Yc
    angle_R = angle_R * math.pi / 180.0
    TOs_O = np.array([[math.cos(angle_R), math.sin(angle_R)], [-math.sin(angle_R), math.cos(angle_R)]])
    # 像平面参考坐标系下 OsXsrYsr，椭圆中心坐标为将像素坐标系转为标准2048的图像坐标系，转为
    Xc, Yc = (Xc - 2048 / 2) * (25/2048), (2048 / 2 - Yc) * (25/2048)
    # print("像平面坐标系坐标",Xc,Yc)
    ellipse_a, ellipse_b = ellipse_a * (25/2048) / 2, ellipse_b * (25/2048) / 2  # 求的半长轴！！！不是全长轴！！！a为短轴，b为长轴
    [Xm, Ym] = TOs_O @ np.array([Xc, Yc])
    # print("像平面参考坐标系坐标", Xm, Ym)
    norm_cfym = math.sqrt(camera_f ** 2 + Ym ** 2)
    rou = (math.atan2(ellipse_b + Xm, norm_cfym) + math.atan2(ellipse_b - Xm, norm_cfym)) / 2
    ne = (math.atan2(ellipse_b + Xm, norm_cfym) - math.atan2(ellipse_b - Xm, norm_cfym)) / 2
    # 由于地心坐标始终位于椭圆的长轴上，即yoer=ym，因此，在像平面参考坐标系下地心坐标可表示为：
    Xoer, Yoer = norm_cfym * math.tan(ne), Ym
    # 将像平面参考坐标系下的地心坐标（xm，ym）转换到像平面坐标系，转换矩阵为：
    TOs_O_ = np.linalg.inv(TOs_O)
    # 像平面坐标系下的地心坐标（xe，ye）可表示为
    [Xe, Ye] = TOs_O_ @ np.array([Xoer, Yoer]).T
    Reds = [camera_f / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            -Xe / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
            Ye / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2)]
    es_dis = r / math.sin(rou)
    obj_core = np.array([Xe/25*2048 + 1024,1024 - Ye/25*2048])
    # print(media_path.split('\\')[-1].split('_')[0:14])
    filename = media_path.split('\\')[-1].split('_')[0:14]
    gt_q = np.array(list(map(float,filename[7:11])))
    utc = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + filename[
        4] + ":" + filename[5] + "." + filename[6]
    time1 = spice.str2et(utc)
    gt_RM1 = quaternion2rot(gt_q)
    T = - gt_RM1 @ (np.array(Reds) * es_dis)
    T_m2c = -T
    alpha = math.atan2(T_m2c[1], T_m2c[0])
    beta = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
    if r < 4000:
        positions2Earth, lightTimes = spice.spkpos('MOON', time1, 'J2000', 'NONE', 'Earth')
        T = positions2Earth + T

    if obj_core is not None:
        # print(obj_core)
        cv2.ellipse(img, ellipse, (0, 5, 205), 2, 2)
        cv2.circle(img, (int(obj_core[0]),int(obj_core[1])), 5, keypoint_color[0], -1)
        cv2.putText(img, f"{keypoint_list[0]}", (int(obj_core[0])+10,int(obj_core[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, keypoint_color[0], 2)
        # cv2.circle(img, (int(ellipse_center[0]),int(ellipse_center[1])), 5, keypoint_color[1], -1)
        # cv2.putText(img, f"{keypoint_list[1]}", (int(ellipse_center[0]),int(ellipse_center[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, keypoint_color[1], 2)

    #
    # if 'snl' in media_path:
    #     cv2.circle(img, (int(img.shape[0]/2),int(img.shape[0]/2)), 5, keypoint_color[2], -1)
    #     cv2.putText(img, f"{keypoint_list[2]}", (int(img.shape[0]/2)-50,int(img.shape[0]/2)+35), cv2.FONT_HERSHEY_SIMPLEX, 2,keypoint_color[2], 2)
    # cv2.imwrite('result/snlem-edimg/{}'.format(media_path.split('\\')[-1]),img)
    return es_dis,bbox_center_ed,ellipse_center,obj_core,T,alpha,beta


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

def create_ExRecord():
    if not os.path.exists("ExRecord_Net.csv"):
        csv_a = open("ExRecord_Net.csv", 'a', encoding='utf-8')
        csv_a.close()
    csv_a = open("ExRecord_Net.csv", 'r', encoding='utf-8')
    if len(csv_a.readlines()) < 1:
        csv_a = open("ExRecord_Net.csv", 'a', encoding='utf-8')
        csv_a.write(
            "实验时间" + "," +
            "模型" + "," +
            "测试集路径" + "," +
            "Resize" + "," +
            "检测成功率" + "," +
            "分类正确率" + "," +
            "不涵盖错误分类NetBBOXC_RMSE" +","+
            "不涵盖错误分类NetEllipseC_RMSE" + "," +
            "不涵盖错误分类NetCore_RMSE" + "," +
            "不涵盖错误分类NetDis_RMSE" + "," +
            "不涵盖错误分类EDEllipseC_RMSE" + "," +
            "不涵盖错误分类EDCore_RMSE" + "," +
            "不涵盖错误分类EDDis_RMSE" + "," +
            "涵盖错误分类NetBBOXC_RMSE" +","+
            "涵盖错误分类NetEllipseC_RMSE" + "," +
            "涵盖错误分类NetCore_RMSE" + "," +
            "涵盖错误分类NetDis_RMSE" + "," +
            "涵盖错误分类EDEllipseC_RMSE" + "," +
            "涵盖错误分类EDCore_RMSE" + "," +
            "涵盖错误分类EDDis_RMSE" +"\n"
        )
        csv_a.close()
    else:
        pass


def plot_err_pixle(time_name,x_lim,suc_err_net_m, suc_err_ed_m,type=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 坐标图像中显示中文
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    if type == 0: # 0:不含错误分类，1：含错误分类
        fig.suptitle('不涵盖错误分类（关于质心提取）', fontsize=20)
    else:
        fig.suptitle('涵盖错误分类（关于质心提取）', fontsize=20)
    from scipy.ndimage import median_filter

    a = median_filter(np.array(suc_err_net_m)[:, 0], 25)
    b = median_filter(np.array(suc_err_net_m)[:, 1], 25)
    c = median_filter(np.array(suc_err_net_m)[:, 2], 25)
    d = median_filter(np.array(suc_err_ed_m)[:, 0], 25)
    e = median_filter(np.array(suc_err_ed_m)[:, 1], 25)
    # a = np.array(suc_err_net_m)[:, 0]
    # b = np.array(suc_err_net_m)[:, 1]
    # c = np.array(suc_err_net_m)[:, 2]
    # d = np.array(suc_err_ed_m)[:, 0]
    # e = np.array(suc_err_ed_m)[:, 1]
    print('bbox_net_err', np.sqrt(np.mean(a ** 2)))
    print('ec_net_err', np.sqrt(np.mean(b ** 2)))
    print('C_net_err', np.sqrt(np.mean(c ** 2)))
    print('ec_ed_err', np.sqrt(np.mean(d ** 2)))
    print('C_ed_err', np.sqrt(np.mean(e ** 2)))
    # ax[0].plot(x_lim,np.array(suc_err_net_m)[:, 0], label='Netbbox_center_err')
    # ax[0].plot(x_lim,np.array(suc_err_net_m)[:, 1], label='Netellipse_center_err')
    ax[0].plot(x_lim,np.array(suc_err_net_m)[:, 2], label='1')
    # ax[0].plot(x_lim,np.array(suc_err_ed_m)[:, 0], label='Edellipse_core_err')
    ax[0].plot(x_lim,np.array(suc_err_ed_m)[:, 1], label='5')
    ax[0].set_ylim(0, 40)
    ax[0].set_xlim(0, 1400)
    # ax[0].set_xlabel(time_name,fontsize=20)
    ax[0].set_ylabel('提取误差[pixel]', fontsize=20)
    ax[0].tick_params(axis='x',labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].legend(prop = {'size':16})

    ax[1].plot(x_lim,np.array(suc_err_net_m)[:, 2], label='1')
    # ax[0].plot(x_lim,np.array(suc_err_ed_m)[:, 0], label='Edellipse_core_err')
    ax[1].plot(x_lim,np.array(suc_err_ed_m)[:, 1], label='5')
    # ax[ 1].set_ylim(0, 40)
    ax[1].set_xlabel(time_name,fontsize=20)
    ax[1].set_xlim(0, 1400)
    ax[1].set_ylabel('提取误差[pixel]', fontsize=20)
    ax[1].tick_params(axis='x',labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].legend(prop = {'size':16})

    # ax[0, 1].plot(x_lim,np.array(suc_err_net_m)[:, 0], label='bbox_center')
    # ax[0, 1].set_xlabel(time_name,fontsize=20)
    # ax[0, 1].set_ylabel('提取误差[pixel]', fontsize=20)
    # ax[0, 1].tick_params(axis='x',labelsize=16)
    # ax[0, 1].tick_params(axis='y', labelsize=16)
    # ax[0, 1].legend(prop = {'size':16})

    # ax[1, 0].plot(x_lim,np.array(suc_err_net_m)[:, 1], label='Netellipse_center_err')
    # ax[1, 0].plot(x_lim,np.array(suc_err_ed_m)[:, 0], label='Edellipse_center_err')
    # ax[1, 0].set_xlabel(time_name, fontsize=20)
    # ax[1, 0].set_ylabel('提取误差[pixel]', fontsize=20)
    # ax[1, 0].tick_params(axis='x', labelsize=16)
    # ax[1, 0].tick_params(axis='y', labelsize=16)
    # ax[1, 0].legend(prop={'size': 16})
    #
    # ax[1, 1].plot(x_lim,np.array(suc_err_net_m)[:, 2], label='Netobj_core_err')
    # ax[1, 1].plot(x_lim,np.array(suc_err_ed_m)[:, 1], label='Edobj_core_err')
    # ax[1, 1].set_xlabel(time_name,fontsize=20)
    # ax[1, 1].set_ylabel('提取误差[pixel]', fontsize=20)
    # ax[1, 1].tick_params(axis='x', labelsize=16)
    # ax[1, 1].tick_params(axis='y', labelsize=16)
    # ax[1, 1].legend(prop={'size': 16})
    plt.show()


def plot_err_dis(time_name,x_lim,suc_err_net_m, suc_err_ed_m, type):
    from scipy.ndimage import median_filter

    a = median_filter(np.array(suc_err_net_m)[:, 3], 25)
    b = median_filter(np.array(suc_err_ed_m)[:, 2], 25)

    print('dis_net_err', np.sqrt(np.mean(a ** 2)))
    print('dis_ed_err', np.sqrt(np.mean(b ** 2)))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 坐标图像中显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,10))
    # if type == 0:  # 0:不含错误分类，1：含错误分类
    #     plt.title('不涵盖错误分类（关于距离估计）', fontsize=20)
    # else:
    #     plt.title('涵盖错误分类（关于距离估计）', fontsize=20)
    plt.plot(x_lim,a, label='1')
    plt.plot(x_lim,b, label='5')
    plt.xlabel(time_name, fontsize=20)
    plt.ylabel('距离估计误差[km]', fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(prop={'size': 20})
    plt.ylim(0, 30000)
    plt.xlim(0, 1400)
    plt.show()


if __name__ == '__main__':
    '''
    创建实验记录表
    '''
    # create_ExRecord()
    '''
    各实验项
    '''
    test_time = datetime.datetime.now()
    weight_path = os.path.dirname(os.path.realpath(__file__)) + "/pt/em.pt" # 权重文件路径
    print("123456789")
    model = YOLO(weight_path)
    print("========")   
    media_dir = os.path.dirname(os.path.realpath(__file__)) + "/image/"# 测试图像路径
    media_ = os.listdir(media_dir)
    media_ = sorted(media_,key=natural_sort_key)
    resize = 640
    detection_fail = 0
    cls_fail = 0
    all_err_net_m = []
    suc_err_net_m = []
    all_err_ed_m = []
    suc_err_ed_m = []
    time_list = []
    # csv_save = open("ExRecord_Net.csv", 'a', encoding='utf-8')
    # center_file_name_mid = [s for s in weight_path.split('\\') if 'yolo' in s]
    # center_file_name = media_dir.split('\\')[-1] + "_" + center_file_name_mid[0] + "_"+str(resize) +"_"+ str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + ".txt"
    # center_file = open('testData/' + center_file_name, 'a')
    for media_path in media_:
        # core_gt = None
        start_time = time.time()
        media_path = os.path.join(media_dir, media_path)
        id, bbox_center_net, ellipse_center_net, obj_core_net, rad_ = net_pre(model, media_path, resize=resize)
        print(time.time() - start_time)
    #     if id is None:
    #         detection_fail += 1  # 检测失败则不进行任何操作，直接跳过，也不计入任何误差计算
    #     else:  # 检测失败则不录入分类失败中
    #         if ('moon' in media_path.split('\\')[-1] and id == 1) or (
    #                 'earth' in media_path.split('\\')[-1] and id == 0):
    #             cls_fail += 1
    #             '''
    #             err_net 存储顺序 bbox_center_err_pixel, ellipse_center_err_pixle, core_err_pixle, es_dis_err
    #             err_ed 存储顺序 ellipse_center_err_pixle, core_err_pixle, es_dis_err
    #             '''
    #         '''
    #         此文件先记录涵盖错误分类的网络和椭圆出的结果，包含位置和测角信息，然后再记录不涵盖错误分类情况下出的所有结果
    #         在对EKF进行输入处理时可以分开进行分析，并且通过media_path 和 id 可有效判断分类问题，无需记录检测率和分类正确率
    #         以及这个文件也可以为画网络与椭圆提取的质心误差、距离误差做输入
    #         '''
    #         center_file.write(str(media_path) + ' ' +
    #                           str(id) + ' ' +
    #                           str(bbox_center_net[0]) + ' ' +
    #                           str(bbox_center_net[1]) + ' ' +
    #                           str(ellipse_center_net[0]) + ' ' +
    #                           str(ellipse_center_net[1]) + ' ' +
    #                           str(obj_core_net[0]) + ' ' +
    #                           str(obj_core_net[1]) + ' ' +
    #                           str(rad_) + ' ')


    #         def obt_err_net_ed(r, core_gt):
    #             filename = media_path.split('\\')[-1].split('_')[0:14]
    #             gt_q = np.array(list(map(float, filename[7:11])))
    #             gt_RM1 = quaternion2rot(gt_q)
    #             err_net = []
    #             err_ed = []
    #             es_dis_ed, bbox_center_ed, ellipse_center_ed, obj_core_ed, T_ed, alpha_ed, beta_ed = ellipse_directly_cal(media_path, r)
    #             if bbox_center_net is not None:
    #                 err_net.append(np.linalg.norm(bbox_center_net - core_gt))
    #                 Xe, Ye = (bbox_center_net[0] - 1024) * 25 / 2048, (1024 - bbox_center_net[1]) * 25 / 2048
    #                 Reds = [camera_f / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
    #                         -Xe / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2),
    #                         Ye / math.sqrt(Xe ** 2 + Ye ** 2 + camera_f ** 2)]
    #             if ellipse_center_net is not None:
    #                 err_net.append(np.linalg.norm(ellipse_center_net - core_gt))
    #             if obj_core_net is not None:
    #                 err_net.append(np.linalg.norm(obj_core_net - core_gt))
    #             if ellipse_center_ed is not None:
    #                 err_ed.append(np.linalg.norm(ellipse_center_ed - core_gt))
    #             if obj_core_ed is not None:
    #                 err_ed.append(np.linalg.norm(obj_core_ed - core_gt))
    #             timestamps = filename[0] + "-" + filename[1] + "-" + filename[2] + "T" + filename[3] + ":" + \
    #                          filename[4] + ":" + filename[5] + "." + filename[6]
    #             timestamps = spice.str2et(timestamps)
    #             time_list.append(timestamps)
    #             if r < 4000:
    #                 xyz_moon, lighttimem = spice.spkpos('10002', timestamps, 'J2000', 'NONE', 'Moon')
    #                 gt_dis = np.linalg.norm(xyz_moon)
    #                 es_dis_net = r / math.sin(rad_)
    #                 T_net = - gt_RM1 @ (np.array(Reds) * es_dis_net)
    #                 T_m2c = -T_net
    #                 alpha_net = math.atan2(T_m2c[1], T_m2c[0])
    #                 beta_net = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
    #                 T_net = xyz_moon + T_net  # 最后都要归到ECI系下
    #                 err_net.append(es_dis_net - gt_dis)
    #                 err_ed.append(es_dis_ed - gt_dis)
    #                 '''
    #                 center_file 再记录上T2ECI、rho、alpha、beta加上椭圆的各项
    #                 '''
    #                 center_file.write(str(ellipse_center_ed[0]) + ' ' +
    #                                   str(ellipse_center_ed[1]) + ' ' +
    #                                   str(obj_core_ed[0]) + ' ' +
    #                                   str(obj_core_ed[1]) + ' ' +
    #                                   str(err_net[0]) + ' ' +
    #                                   str(err_net[1]) + ' ' +
    #                                   str(err_net[2]) + ' ' +
    #                                   str(err_net[3]) + ' ' +
    #                                   str(err_ed[0]) + ' ' +
    #                                   str(err_ed[1]) + ' ' +
    #                                   str(err_ed[2]) + ' ')
    #             else:
    #                 xyz_earth = np.array(list(map(float, filename[11:14])))
    #                 gt_dis = np.linalg.norm(xyz_earth)
    #                 es_dis_net = r / math.sin(rad_)
    #                 err_net.append(es_dis_net - gt_dis)
    #                 err_ed.append(es_dis_ed - gt_dis)
    #                 T_net = - gt_RM1 @ (np.array(Reds) * es_dis_net)
    #                 T_m2c = -T_net
    #                 alpha_net = math.atan2(T_m2c[1], T_m2c[0])
    #                 beta_net = math.atan2(T_m2c[2], math.sqrt(T_m2c[0] ** 2 + T_m2c[1] ** 2))
    #                 center_file.write(str(ellipse_center_ed[0]) + ' ' +
    #                                   str(ellipse_center_ed[1]) + ' ' +
    #                                   str(obj_core_ed[0]) + ' ' +
    #                                   str(obj_core_ed[1]) + ' ' +
    #                                   str(err_net[0]) + ' ' +
    #                                   str(err_net[1]) + ' ' +
    #                                   str(err_net[2]) + ' ' +
    #                                   str(err_net[3]) + ' ' +
    #                                   str(err_ed[0]) + ' ' +
    #                                   str(err_ed[1]) + ' ' +
    #                                   str(err_ed[2]) + ' ')
    #             return err_net, err_ed


    #         # 涵盖错误分类情况
    #         r = 1737.40 if id == 0 else 6378.1366
    #         if id == 0:
    #             uv, uv_pixel = moon_calculate(media_path)
    #             core_gt = np.array([uv_pixel[0] + 1024, 1024 - uv_pixel[1]])
    #         else:
    #             uv, uv_pixel = geo_calculate(media_path)
    #             core_gt = np.array([uv_pixel[0] + 1024, 1024 - uv_pixel[1]])
    #         err_net, err_ed = obt_err_net_ed(r, core_gt)
    #         all_err_net_m.append(err_net)
    #         all_err_ed_m.append(err_ed)

    #         # 不涵盖错误分类情况
    #         r = 1737.40 if 'moon' in media_path.split('\\')[-1] or 'Moon' in media_path.split('\\')[-2] or 'Moon' in media_path.split('\\')[-3] else 6378.1366
    #         if 'moon' in media_path.split('\\')[-1] or 'Moon' in media_path.split('\\')[-2] or 'Moon' in media_path.split('\\')[-3] :
    #             uv, uv_pixel = moon_calculate(media_path)
    #             core_gt = np.array([uv_pixel[0] + 1024, 1024 - uv_pixel[1]])
    #         else:
    #             uv, uv_pixel = geo_calculate(media_path)
    #             core_gt = np.array([uv_pixel[0] + 1024, 1024 - uv_pixel[1]])
    #         err_net, err_ed = obt_err_net_ed(r, core_gt)
    #         suc_err_net_m.append(err_net)
    #         suc_err_ed_m.append(err_ed)
    #         center_file.write('\n')
    # center_file.close()
    '''
    处理X轴刻度坐标将其转化成从0开始的sec、min or h
    '''
    # time_list = np.array(time_list[::2])
    # x_lim_sec = time_list - time_list[0]
    # x_lim_min = x_lim_sec/60
    # x_lim_h = x_lim_min/60
    # x_lim_day = x_lim_h/24
    # x_lim = x_lim_sec
    # time_name = '时间[s]'
    # print("检测成功率：", str(1 - detection_fail / len(media_)))
    # print("分类成功率：", str(1 - cls_fail / (len(media_) - detection_fail)))
    # print("不涵盖错误分类NetBBOX_center_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 0] ** 2))))
    # print("不涵盖错误分类NetEllipse_center_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 1] ** 2))))
    # print("不涵盖错误分类Netobj_core_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 2] ** 2))))
    # print("不涵盖错误分类Net_dis_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 3] ** 2))))
    # print("---------------------------------------")
    # print("不涵盖错误分类EDEllipse_center_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 0] ** 2))))
    # print("不涵盖错误分类EDobj_core_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 1] ** 2))))
    # print("不涵盖错误分类EDes_dis_RMSE:", str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 2] ** 2))))
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print("涵盖错误分类NetBBOX_center_RMSE:", str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 0] ** 2))))
    # print("涵盖错误分类NetEllipse_center_RMSE:", str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 1] ** 2))))
    # print("涵盖错误分类Netobj_core_RMSE:", str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 2] ** 2))))
    # print("涵盖错误分类Net_dis_RMSE:", str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 3] ** 2))))
    # print("---------------------------------------")
    # print("涵盖错误分类EDEllipse_center_RMSE:", str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 0] ** 2))))
    # print("涵盖错误分类EDobj_core_RMSE:", str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 1] ** 2))))
    # print("涵盖错误分类EDes_dis_RMSE:", str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 2] ** 2))))
    # csv_save.write(str(test_time) + "," +
    #                str(center_file_name_mid) + "," +
    #                str(media_dir) + "," +
    #                str(resize) + "," +
    #                str(1 - detection_fail / len(media_)) + "," +
    #                str(1 - cls_fail / (len(media_) - detection_fail)) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 0] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 1] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 2] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_net_m)[:, 3] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 0] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 1] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(suc_err_ed_m)[:, 2] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 0] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 1] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 2] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_net_m)[:, 3] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 0] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 1] ** 2))) + "," +
    #                str(np.sqrt(np.mean(np.array(all_err_ed_m)[:, 2] ** 2))) + "\n")
    # csv_save.close()
    # ################################################画图
    # plot_err_pixle(time_name,x_lim,suc_err_net_m, suc_err_ed_m, type=0)  # 0:不含错误，1：含错误
    # plot_err_pixle(time_name,x_lim,all_err_net_m, all_err_ed_m, type=1)
    # plot_err_dis(time_name,x_lim,suc_err_net_m, suc_err_ed_m, type=0)
    # plot_err_dis(time_name,x_lim,all_err_net_m, all_err_ed_m, type=1)
