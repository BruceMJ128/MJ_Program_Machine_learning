# -*- coding: utf-8 -*-
import numpy as np
from math import *
from numpy import *

import numpy

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')   #原始数据testSet的数据格式为：数据1，数据2，label
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #其中的1.0为迭代的初始值
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

   
            
#L(w)=∑[yi*(xi*w)-log(1+exp(xi*w))]，求该函数的最小值
def quasi_Newton_DFP(dataMatIn, classLabels, numIter):
    dataMatrix = mat(dataMatIn)             #X: convert to numpy.matrixlib.defmatrix.matrix
    labelMat = mat(classLabels).transpose() #Y: convert to NumPy matrix，再转置
    m,n = shape(dataMatrix)   
    wk = mat(ones((n,1)))                   #wk初始化为[1,1,1]      
    
    # 计算初始二阶导数（海赛矩阵）                       #∂2L/∂wi∂wj = ∑[(-1)*x_ki*x_kj*exp(x[k]*w)*(1+exp(x[k]*w))^-2]
    
    H0 = mat(eye(n,M=None,k=0))
    for i in range(n):
        for j in range(n):
            H0[i,j]=0
            for k in range(m):
                H0[i,j]=H0[i,j]+(dataMatrix[k,i]*dataMatrix[k,j]*exp(dataMatrix[k]*wk))/(1+exp(dataMatrix[k]*wk))
    Gk = numpy.linalg.inv(H0)           #Gk：用H0的逆矩阵初始化G0，G0 ~ H0^-1
    '''
    a, b = numpy.linalg.eig(H0)         #特征值与特征向量，a为特征值 向量，b为特征向量矩阵
    c=np.empty((n), dtype=float)    
    for i in range(n):
        c[i] = 1/a[i]    
    Gk =numpy.diag(c)                   #Gk：用H0的特征值的导数作为特征值初始化G0，G0 ~ H0^-1
    '''
    #一阶导数初值 g0                   #∂L/∂wi=∑ [xki*yk + xki/(1+exp(-x[k]*w) - xki]
    
    gk = mat(zeros((n,1)))  
    
    for i in range(n):                          #计算gk+1
        for k in range(m):
            gk[i] = gk[i] + dataMatrix[k,i]*labelMat[k]+ dataMatrix[k,i]/(1+exp(dataMatrix[k]*wk) - dataMatrix[k,i])
        
    norm_gk = math.sqrt(float(gk.transpose()*gk))  #向量gk的范数
    
    La =0                   #λ(Lambda), f(xk+λ*pk)
    L =0                    #L(w)=∑[yi*(xi*w)-log(1+exp(xi*w))]
    
    for iv in range(numIter):                     #最多迭代numIter次
        if norm_gk < 10^-8:
            break        
        pk = -Gk*gk
        La_min = 0
        Lmin = 0   #min f(xk + λ*pk): Lmin
        L = 0                                 
        for l in range(20):
             La = float(l/10.0)           #λ(Lambda), 在0~1.9之间找一个值，使 f(xk + λ*pk)最小
             wk_temp = wk + La*pk
             for ii in range(m):
                 L = L + labelMat[ii]*(dataMatrix[ii]*wk_temp) - math.log(1+math.exp(dataMatrix[ii]*wk_temp))  #F: f(xk+λ*pk), 即 L(w)=∑[yi*(xi*w)-log(1+exp(xi*w))]
             if L<Lmin:
                 Lmin = L
                 La_min = La
                 
        wk = wk + La_min*pk
        dk = La_min*pk                         #δk Delta k
        
        z = dataMatrix * wk
        gk_old = gk 
        for i in range(n):                          #计算gk+1
            x_mi= mat([x[i] for x in dataMatIn])
            SH = 0 
            for j in range(m):
                SH = SH + (-1)*(exp(z[j]*dataMatrix[j,i]))/(1+exp(z[j]))  #Second half: - 1/(1+exp(-(x*w)))
            gk[i] = x_mi*labelMat + SH          
        norm_gk = math.sqrt(float(gk.transpose()*gk))
        yk = gk - gk_old                        # yk = gk+1 - gk
        
        Gk = Gk+(dk*dk.transpose())/(dk.transpose()*yk)-(Gk*yk*yk.transpose()*Gk)/(yk.transpose()*Gk*yk)    
    return wk
    
dataMatIn, classLabels = loadDataSet()
numIter = 50
weights = quasi_Newton_DFP(dataMatIn, classLabels, numIter)
    
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
plotBestFit(weights.getA())    