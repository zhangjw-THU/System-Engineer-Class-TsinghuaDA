# -*- coding: utf-8 -*-
# 张嘉玮
# 20190423

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as scio
import datetime

def center_dis(data,center):
    """
    计算data[i]到center[j]的距离
    :param data: 原始数据
    :param center: 聚类中心
    :return: 距离矩阵
    """
    dis = []
    for i,x in zip(range(len(data)),data):
        dis.append([])
        for c in center:
            dis[i].append(np.linalg.norm(c-x))

    return np.asarray(dis)

def assign_data(dis):
    """
    根据距离，重新划分数据
    :param dis: 距离举证
    :return: 划分好的数据
    """

    assignment = []
    for d in dis:
        d = list(d)
        assignment.append(d.index(min(d)))

    return np.array(assignment)

def update_centers(data,assignment,num):
    """
    更新聚类中心
    :param data: 原始数据
    :param assignment: 新的类别
    :param num: 聚类数
    :return: 新的聚类中心
    """

    data_clusters = [[] for _ in range(num)]
    for i,x in zip(assignment,data):
        data_clusters[i].append(x)

    centers = [np.mean(data_cluster,axis=0) for data_cluster in data_clusters]

    return centers

def computer_err(data,assignment,centers):
    """
    计算目标函数值，也就是中的cost
    :param data: 原始数据
    :param assignment: 数据所属聚类
    :param centers: 聚类中心
    :return: 总的loss
    """
    err = 0
    for i,x in zip(assignment,data):
        err = err + np.linalg.norm(x-centers[i])
    err = err/len(data)
    return err

def gameover(centers,new_centers):
    """
    判断是否收敛
    :param centers: 上一步的centers
    :param new_centers: 这一步的centers
    :return:
    """
    ERROR = 1e-4
    now = 0
    for old ,new in zip(centers,new_centers):
        now = now + np.linalg.norm(old-new)

    if now<ERROR:
        return True
    else:
        return False

def plot_2D(data,assigment,num):
    """
    若数二维数据，可视化
    :param data: 原始数据
    :param assigment: 对应聚类
    :param num: 聚类数
    :return:
    """
    scatterColors = ['orange','purple',  'brown','black', 'blue', 'green', 'yellow', 'red' ]
    for i in range(num):
        color = scatterColors[i%len(scatterColors)]
        x = []
        y = []
        for j in range(len(data)):
            if assigment[j] == i:
                x.append(data[j,0])
                y.append(data[j,1])
        plt.scatter(x,y,c= color,alpha=1,marker='x')

    plt.show()


def Kmeans_clustering(data,num):
    """
    K-means 聚类
    :param data: 原始数据：N*M
    :param num: 聚类数
    :return: cost
    """

    centers = []
    # random.seed(423)  #若添加种子，则每次聚类结果一样
    for i in range(num):
        # 随机初始化
        centers.append(data[random.randint(0,len(data))])
        # 指定初始化，用前num个数据进行初始化
        # centers.append(data[i])

    Iteration = 0
    while 1:
        Iteration += 1
        dis = center_dis(data,centers)
        assigments = assign_data(dis)
        new_centers = update_centers(data,assigments,num)

        if gameover(centers,new_centers):
            centers = new_centers
            err = computer_err(data, assigments, centers)
            # print("ITERATION:", Iteration, "    LOSS:", err)
            break
        else:
            centers = new_centers

        # err = computer_err(data,assigments,centers)
        # if Iteration%3==0:
        #     print("ITERATION:",Iteration,"    LOSS:",err)

    if len(data[0])==2:
        plot_2D(data,assigments,num)
    return round(err,3)

# 载入数据
path = 'data.mat'
datas = scio.loadmat(path)['data']
Kmeans_clustering(datas,3)

############# 确定K的大小 ###################################################################
# err = []
# Kmin = 2
# Kmax = 20
# for i in range(Kmin,Kmax):
#
#     print("K = ",i)
#     err.append(Kmeans_clustering(data, 8))
#
# x = range(Kmin,Kmax)
# plt.title('Cost Test')
# plt.plot(x,err,label='Coct', linewidth=3, color='r', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.xlabel('K=8')
# plt.ylabel('Cost')
# for a,b in zip(x,err):
#     plt.text(a,b,b,ha='center', va='bottom', fontsize=10)
# plt.show()


############# 统计时间 #####################################
# ！！！！！为了使得时间具有可比性，修改上面的初始化，使得前num为初始值！！！！！
# T = []
# for i in range(1,16):
#     print(i,"    datasets: " , i*200)
#     starttime = datetime.datetime.now()
#     Kmeans_clustering(datas[0:i*200-1], 4)
#     endtime = datetime.datetime.now()
#     T.append((endtime-starttime).microseconds)
#     print(endtime-starttime)
#
# x = [i*200 for i in range(1,16)]
# plt.title('Time Cost')
# plt.plot(x,T,label='Cost', linewidth=3, color='r', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.xlabel('datasets')
# plt.ylabel('Cost(ns)')
# for a,b in zip(x,T):
#     plt.text(a,b,b,ha='center', va='bottom', fontsize=10)
# plt.show()