# -*- coding: utf-8 -*-
# 张嘉玮
# 20190423

函数说明：

def center_dis(data,center):
    """
    计算data[i]到center[j]的距离
    :param data: 原始数据
    :param center: 聚类中心
    :return: 距离矩阵
    """
	
def assign_data(dis):
    """
    根据距离，重新划分数据
    :param dis: 距离举证
    :return: 划分好的数据
    """
	
def update_centers(data,assignment,num):
    """
    更新聚类中心
    :param data: 原始数据
    :param assignment: 新的类别
    :param num: 聚类数
    :return: 新的聚类中心
    """
	
def computer_err(data,assignment,centers):
    """
    计算目标函数值，也就是中的cost
    :param data: 原始数据
    :param assignment: 数据所属聚类
    :param centers: 聚类中心
    :return: 总的loss
    """
	
def gameover(centers,new_centers):
    """
    判断是否收敛
    :param centers: 上一步的centers
    :param new_centers: 这一步的centers
    :return:
    """
	
def plot_2D(data,assigment,num):
    """
    若数二维数据，可视化
    :param data: 原始数据
    :param assigment: 对应聚类
    :param num: 聚类数
    :return: 
    """

def Kmeans_clustering(data,num):
    """
    K-means 聚类
    :param data: 原始数据：N*M
    :param num: 聚类数
    :return: cost
    """
	
	