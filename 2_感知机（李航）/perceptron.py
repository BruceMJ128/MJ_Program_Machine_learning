# -*- coding: utf-8 -*-
import numpy as np
from numpy import *

def loadDataSet():
    dataMat=[]; labelMat =[]
    fr = open('perceptron_data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  #将一行内容拆分，转换为list
        dataMat.append([int(lineArr[0]),int(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sign(x, w, b):
    if (w*x+b)<0:
        return -1
    else:
        return 1

def perceptron_original(dataArr, labelArr, eta):  #eta, η, 为迭代步长
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()  
    m,n=shape(dataMat)
    w = mat(zeros(n))
    b=0
    maxcircles = 100  #最大循环次数
    z = zeros(m)  #判断各行yi*(w*xi+b)是否大于0，若大于0，则赋值1；否则赋值0
    while(sum(z)<m and maxcircles >0):
        for i in range(m):
            if (labelMat[i]*(w*dataMat[i].transpose()+b))<=0:
                z[i]=0
                w = w + eta*labelMat[i]*dataMat[i]
                b = b + eta*labelMat[i]
                print w,b
            else:
                z[i]=1
        maxcircles = maxcircles -1
        
    return w,b  

'''
dataArr,labelArr = loadDataSet()
m, n = shape(dataArr)
#w = empty(m)
eta =1 
w0, b0 = perceptron_original(dataArr, labelArr, eta)  #取迭代步长为1
'''
def perceptron_dualization(dataArr, labelArr, eta):
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()  
    m,n=shape(dataMat)
    a = mat(zeros(m)).transpose()   #即alpha, α
    b=0
    maxcircles = 100  #最大循环次数
    z = zeros(m)  #判断各行yi*(∑（a*yj*xj）*xi+b)是否大于0，若大于0，则赋值1；否则赋值0    
    
    
    while(sum(z)<m and maxcircles >0):
        for i in range(m):
            x = mat(zeros(n))             #初始化aj*yj*xj
            for j in range(m):
                x = x + a[j]*labelArr[j]*dataMat[j]
            if (labelMat[i]*(x*dataMat[i].transpose()+b))<=0:
                z[i]=0
                a[i] = a[i] + eta 
                b = b + eta*labelMat[i]
                print 'a[',i,']:',a[i],'; b:',b
            else:
                z[i]=1
        maxcircles = maxcircles -1        
    return a,b  
    
dataArr,labelArr = loadDataSet()
m, n = shape(dataArr)
#w = empty(m)
eta =1 
a1, b1 = perceptron_dualization(dataArr, labelArr, eta)  #取迭代步长为1