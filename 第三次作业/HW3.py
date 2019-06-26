# -*- coding: utf-8 -*-
# 张嘉玮
# 20190403
import numpy as np
from scipy.stats import f
from scipy.stats import norm
import matplotlib.pyplot as plt

def linear_regression1(data,alpha):
    N = len(data)
    Y = np.array([data[i][0] for i in range(N)])
    X = np.array([data[i][1] for i in range(N)])


    Y_ = np.mean(Y)
    X_ =  np.mean(X)

    L_xx = np.sum([i*i for i in X])-np.square(np.sum(X))/N
    L_xy = np.sum(np.multiply(X,Y)) - np.sum(X)*np.sum(Y)/N
    L_yy = np.sum([j*j for j in Y]) - np.square(np.sum(Y))/N

    b = L_xy/L_xx
    a = Y_ - b*X_

    # 打印出回归直线的方程
    plt.figure(1)
    plt.plot(X,Y,'yo')
    Y_pre = np.array([b*i+a for i in X])
    plt.plot(X,Y_pre,'r')
    Text = 'RegressionLinear: y = '+str(round(b,3))+'x+'+str(round(a,3))
    plt.text(min(X), min(Y_pre), Text, wrap=True)
    plt.title(u'线性回归', fontproperties='SimHei', fontsize=14)
    plt.xlabel(u'X', fontproperties='SimHei', fontsize=14)
    plt.ylabel(u'Y', fontproperties='SimHei', fontsize=14)
    plt.show()

    #F检验：根据F表，获得临界值
    F = f.ppf(alpha,1,N-2)
    U = b*L_xy
    Q = L_yy - U
    F_ = U*(N-2)/Q

    # 满足线性关系
    if F_ > F:
        print('   在F检验下，满足线性关系')

        S_a = np.sqrt(np.sum([i*i for i in Y-Y_pre])/(N-2))
        sig = norm.ppf(1-alpha/2)
        Y_max = Y_pre+sig*S_a
        Y_min = Y_pre-sig*S_a
        print('   置信区间（误差范围） ：[',-sig*S_a,',',sig*S_a,']')
        plt.figure(2)
        Text = 'RegressionLinear: y = ' + str(round(b, 3)) + 'x+' + str(round(a, 3))
        plt.text(min(X), min(Y_pre), Text, wrap=True)
        plt.plot(X,Y_pre,'r')
        plt.plot(X,Y_max,'b')
        plt.plot(X,Y_min,'b')
        plt.plot(X,Y,'yo')
        plt.title(u'线性回归',fontproperties='SimHei', fontsize=14)
        plt.xlabel(u'X', fontproperties='SimHei', fontsize=14)
        plt.ylabel(u'Y', fontproperties='SimHei', fontsize=14)
        plt.show()

    # 不满足线性关系
    else:
        print('   在F检验下，不满足线性关系')


data= [[4,0.009],[3.44,0.013],[3.6,0.006],[1.0,0.025],[2.04,0.022],[4.47,0.007],[0.6,0.036],[1.7,0.014],[2.92,0.016],[4.8,0.014],[3.28,0.016],[4.16,0.012],[3.35,0.020],[2.2,0.018]]
linear_regression1(data,0.05)