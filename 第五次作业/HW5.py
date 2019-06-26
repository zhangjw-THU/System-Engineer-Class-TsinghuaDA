# -*- coding: utf-8 -*-
# 张嘉玮
# 20190419

import numpy as np
import pandas as pd

from scipy.stats import f
from scipy.stats import norm

def pca_compress(data,rerr):
    """
    PCA
    :param data:输入的原始数据矩阵，每一行对应一个数据点
    :param rerr:相对误差界限，即相对误差应当小于这个值，用于确定主成分个数
    :return:各个主成分，每一列为一个主成分pcs;压缩后的数据，每一行对应一个数据点;压缩时的一些常数，包括数据每一维的均值和方差等。利用以上三个变量应当可以恢复出原始的数据
    """
    data_dim = len(data[0])
    data = np.array(data)
    data_T = np.transpose(data)
    data_num = len(data_T[0])

    # 样本数据标准化
    means = np.zeros(data_dim)
    vars = np.zeros(data_dim)
    cprs_c = []
    for i in range(data_dim):
        means[i] = np.mean(data_T[i])
        vars[i] = np.var(data_T[i])
        cprs_c.append([means[i],vars[i]])
        data_T[i] = (data_T[i]-means[i])/np.sqrt(vars[i])

    # 计算协方差矩阵
    X_T = np.transpose(data_T)
    X = data_T
    XX_T = np.dot(X,X_T)

    # 特征值和特征向量
    eigenvalue, featurevector = np.linalg.eig(XX_T)
    index_sorted = np.argsort(-eigenvalue)

    # 对特征值进行排序
    eigenvalue_sorted = eigenvalue[index_sorted]
    featurevector_T = np.transpose(featurevector)
    featurevector_T_sorted = featurevector_T[index_sorted]
    featurevector_sorted = np.transpose(featurevector_T_sorted)

    # print("特征值：",eigenvalue_sorted)
    # 特征值个数
    m = data_num
    for i in range(data_num-1,0,-1):
        if np.sum(eigenvalue_sorted[i:])/np.sum(eigenvalue_sorted) >rerr:
            m = i+1
            break

    # print("使用特征值的个数m:",m)
    # 得到前m个特征值
    featurevector_sorted_T = np.transpose(featurevector_sorted)
    pcs_T = featurevector_sorted_T[0:m]
    pcs = np.transpose(pcs_T)

    # 计算投影
    cprs_data_T = np.dot(pcs_T,X)
    cprs_data = np.transpose(cprs_data_T)
    # 返回值
    return pcs,cprs_data,cprs_c


def pca_reconstruct(pcs,cprs_data,cprs_c):
    """
    DPCA
    数据恢复
    :param pcs: 各个主成分，每一列为一个主成分
    :param cprs_data:压缩后的数据，每一行对应一个数据点
    :param cprs_c:压缩时的一些常数，包括数据每一维的均值和方差等。利用以上三个变量应当可以恢复出原始的数据
    :return:恢复出来的数据，每一行对应一个数据点
    """
    X = np.transpose(cprs_data)
    recon_data_one = np.dot(pcs,X)
    X_dim = recon_data_one.shape[0]
    recon_data_two = recon_data_one
    for i in range(X_dim):
        recon_data_one[i] = recon_data_one[i]*np.sqrt(cprs_c[i][1])+cprs_c[i][0]

    return np.transpose(recon_data_two)


def linear_regression(Y,X,alpha):
    """
    可自适应病态问题的多元线性回归问题
    :param Y: 1xN
    :param X: NxX的维度
    :param alpha: 显著性检验参数
    :return: 无
    """
    x_dim = len(X[0])
    N = len(Y)

    Y = np.array(Y)
    X = np.array(X)

    # X 归一化
    x_means = np.array([np.mean(X[:,i]) for i in range(x_dim)])
    x_vars = np.array([np.var(X[:,i]) for i in range(x_dim)])
    X_new = np.zeros((N,x_dim))
    for i in range(N):
        for j in range(x_dim):
            X_new[i][j] = (X[i][j]-x_means[j])/np.sqrt(x_vars[j])

    # Y 归一化
    y_mean = np.mean(Y)
    y_var = np.var(Y)
    Y_new = np.array([[(i-y_mean)/np.sqrt(y_var) for i in Y]])

    Xt = np.transpose(X_new)
    XXt = np.dot(Xt,X_new)

    # 求得特征值和特征向量
    eigenvalue, featurevector = np.linalg.eig(XXt)
    # 排序
    eigenvalue_index_sort = np.argsort(eigenvalue)
    m = len(eigenvalue)
    for i in range(len(eigenvalue)):
        idx = eigenvalue_index_sort[0:i+1]
        if np.sum([eigenvalue[j] for j in idx])/np.sum(eigenvalue)>0.1:
            m = m-i
            break

    # 病态与否判断
    if m!=len(eigenvalue):
        print("[1] 经检验，该线性回归问题为病态线性回归问题")
    else:
        print("[1] 经检验，该线性回归问题没有病态")

    Q_m = []
    featurevector.tolist()
    for i in range(m):
        a = featurevector[:,eigenvalue_index_sort[len(eigenvalue)-1-i]]
        Q_m.append(a.tolist())
    Q_m = np.array(Q_m)
    Q_m = np.transpose(Q_m)
    ZZ_T_inverse = np.zeros((m,m))

    for i in range(m):
        ZZ_T_inverse[i][i] = 1/eigenvalue[eigenvalue_index_sort[len(eigenvalue)-1-i]]
    d = np.dot(np.dot(ZZ_T_inverse,np.transpose(Q_m)),np.dot(np.transpose(X_new),np.transpose(Y_new)))
    c_0 = np.dot(Q_m,d)

    # 去规范化：
    c = [c_0[i]*np.sqrt(y_var)/np.sqrt(x_vars[i]) for i in range(x_dim)]
    bias = y_mean-np.sum([x_means[i]*c[i] for i in range(x_dim)])

    # 打印输出回归结果
    EffectiveNumber = 6
    S = 'Z = '
    for i in range(x_dim):
        if c[i][0]>0:
            S = S+str(round(c[i][0],EffectiveNumber))+'*Y'+str(i+1) + ' + '
        elif c[i][0]<0:
            S = S+'('+str(round(c[i][0],EffectiveNumber))+')*Y'+str(i+1) + ' + '
    if bias>0:
        S = S+str(round(bias,EffectiveNumber))
    elif bias<0:
        S = S+'('+str(round(bias,EffectiveNumber))+')'
    print("[2] 线性回归方程为：",S)

    # 显著性检验：
    Y_pre = []
    for i in range(N):
        Y_pre.append(bias+np.sum([c[j][0]*X[i][j] for j in range(x_dim)] ))
    ESS = np.sum([np.square(Y_pre[i]-y_mean) for i in range(N)])
    RSS = np.sum([np.square(Y_pre[i]-Y[i]) for i in range(N)])
    F = (ESS/x_dim)/(RSS/(N-x_dim-1))
    F_0 = f.ppf(1-alpha,x_dim,N-x_dim-1)
    print("[3] F检验的值：",F,"     F检验的临界值：",F_0)
    if F>F_0:
        print("[4] 在显著水平取：",alpha,"时，经检验，线性相关")
    else:
        print("[4] 在显著水平取：",alpha,"时，经检验，线性无关")
    sig = norm.ppf(1 - alpha / 2)
    S_a = np.sqrt(np.sum([i * i for i in Y - Y_pre]) / (N - x_dim - 1))
    print('[5] 置信区间（误差范围） ：[', -sig * S_a, ',', sig * S_a, ']')

    return c,bias

def PrintModel(pcs,cprs_c,c,bias,Y_mean,Y_var):
    """
    打印最终模型
    :param pcs:各个主成分，每一列为一个主成分
    :param cprs_c:压缩时的一些常数，包括数据每一维的均值和方差等。利用以上三个变量应当可以恢复出原始的数据
    :param c:降维模型的系数
    :param bias:降维模型的偏执
    :return:原数据模型的系数
    """
    c = np.transpose(c)
    X_dim = len(pcs)
    pcs_T = np.transpose(pcs)  # 一行对应一个特征值
    c_new = np.dot(c,pcs_T)[0]
    bias_new = bias
    for i in range(X_dim):
        c_new[i] = c_new[i]/np.sqrt(cprs_c[i][1])
        bias_new = bias_new-c_new[i]*cprs_c[i][0]

    modulus = c_new*np.sqrt(Y_var)
    Bias = bias_new*np.sqrt(Y_var)+Y_mean
    # 打印输出回归结果
    EffectiveNumber = 9
    S = 'Turnout = '
    for i in range(X_dim):
        if modulus[i] > 0:
            S = S + str(round(modulus[i], EffectiveNumber)) + '*X' + str(i + 1) + ' + '
        elif modulus[i] < 0:
            S = S + '(' + str(round(modulus[i], EffectiveNumber)) + ')*X' + str(i + 1) + ' + '
    if Bias > 0:
        S = S + str(round(Bias, EffectiveNumber))
    elif Bias < 0:
        S = S + '(' + str(round(Bias, EffectiveNumber)) + ')'
    print("[*] 线性回归方程为：", S)

    return modulus,Bias

print("####################################################################################################")
PRINTNUM = 1
# 读取数据
data_row = pd.read_excel('counties.xlsx',usecols='C:P',type=float)
Y_row = pd.read_excel('counties.xlsx',usecols='Q',type=float)
data = data_row.as_matrix()
Y = Y_row.as_matrix()

# 压缩
pcs,cprs_data,cprs_c=pca_compress(data,0.10)

print("【",PRINTNUM,"】检验原始数据是否线性相关：\n")
PRINTNUM = PRINTNUM+1
# 检验原始数据
linear_regression(np.transpose(Y)[0],data,0.05)

print("####################################################################################################")
print("【",PRINTNUM,"】各个主成分（每一列一个主成分）：\n",pcs)
PRINTNUM = PRINTNUM+1
print("####################################################################################################")
print("【",PRINTNUM,"】压缩后的数据（每一行对应一个数据）：\n",cprs_data)
PRINTNUM = PRINTNUM+1
print("####################################################################################################")
print("【",PRINTNUM,"】压缩后的维度：\n",len(cprs_data[0]))
PRINTNUM = PRINTNUM+1
print("####################################################################################################")
print("【",PRINTNUM,"】压缩后的数据大小：\n",np.shape(cprs_data))
PRINTNUM = PRINTNUM+1
print("####################################################################################################")
print("【",PRINTNUM,"】原始数据的均值方差：\n",cprs_c)
PRINTNUM = PRINTNUM+1
# 解压缩
recon_data = pca_reconstruct(pcs,cprs_data,cprs_c)

print("####################################################################################################")
print("【",PRINTNUM,"】对压缩数据进行线性回归：\n")
PRINTNUM = PRINTNUM+1
# 回归
## 【方法】：先对Y进行归一化，再回归
Y = np.transpose(Y)[0]
Y_mean = np.mean(Y)
Y_var = np.var(Y)
Y_inl = (Y-Y_mean)/np.sqrt(Y_var)
c,bias = linear_regression(Y_inl,cprs_data,0.05)


print("####################################################################################################")
print("【",PRINTNUM,"】去归一化：\n")
PRINTNUM = PRINTNUM+1
PrintModel(pcs,cprs_c,c,bias,Y_mean,Y_var)
