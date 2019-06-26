# -*- coding: utf-8 -*-
# 张嘉玮
# 20190409
import numpy as np
from scipy.stats import f
from scipy.stats import norm
import pysnooper

@pysnooper.snoop()
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
    S = 'y = '
    for i in range(x_dim):
        if c[i][0]>0:
            S = S+str(round(c[i][0],3))+'*X'+str(i+1) + ' + '
        elif c[i][0]<0:
            S = S+'('+str(round(c[i][0],3))+')*X'+str(i+1) + ' + '
    if bias>0:
        S = S+str(round(bias,3))
    elif bias<0:
        S = S+'('+str(round(bias,3))+')'
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

# 病态问题
Y = [15.9,16.4,19.0,19.1,18.88,20.4,22.7,26.5,28.1,27.6,26.3]
X = [[149.3,4.2 ,80.3 ,108.1],
     [161.2,4.1 ,72.9 ,114.8],
     [171.5,3.1 ,45.6 ,123.2],
     [175.5,3.1 ,50.2 ,126.9],
     [180.8,1.1 ,68.8 ,132.0],
     [190.7,2.2 ,88.5 ,137.7],
     [202.1,2.1 ,87.0 ,146.0],
     [212.4,5.6 ,96.9 ,154.1],
     [226.1,5.0 ,84.9 ,162.3],
     [231.9,5.1 ,60.7 ,164.3],
     [239.0,0.7 ,70.4 ,167.6]]
linear_regression(Y,X,0.05)

# 正常问题
X1 = [[100,4],
      [50,3],
      [100,4],
      [100,2],
      [50,2],
      [80,2],
      [75,3],
      [65,4],
      [90,3],
      [90,2]]
Y1 =[9.3, 4.8, 8.9, 6.5, 4.2, 6.2, 7.4, 6, 7.6, 6.1]
linear_regression(Y1,X1,0.05)
