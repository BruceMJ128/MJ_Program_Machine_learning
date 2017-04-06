# -*- coding: utf-8 -*-
from numpy import *

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

#利用已有数据训练模型，目的是得到 weight向量 w = [w1, w2, ..., wn], 使得 dataMatIn * weights = classLabels

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to numpy.matrixlib.defmatrix.matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix，再转置
    m,n = shape(dataMatrix)
    alpha = 0.001                           #alpha 步长
    maxCycles = 500                         #maxCycles 迭代次数
    weights = ones((n,1))                   #初始迭代，w向量赋值为 1
    for k in range(maxCycles):              #heavy on matrix operations，跑500次
        h = sigmoid(dataMatrix*weights)     #matrix mult，即z = w0*x0 + w1*x1 + w2*x2 + ... +wn*xn，先用各项乘以系数再加和，再用sigmoid获得类似于0或1的值，进行分类
        error = (labelMat - h)              #vector subtraction, 用上一步计算出来的值，与向量的label相减，为（100*1）的向量
        weights = weights + alpha * dataMatrix.transpose()* error    #matrix mult，用原书数据与error相乘，对w0，w1, w2, ... wn 进行迭代 // error越大权重越大，说明其需要进一步迭代的必要性越大
    return weights

dataArr, labelMat = loadDataSet()

weights = gradAscent(dataArr, labelMat)




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

'''
#随机梯度上升算法

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
	
dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(array(dataArr), labelMat)
plotBestFit(weights)

dataMatrix = array(dataArr)
classLabels = labelMat

m,n = shape(dataMatrix) #m=100, n=3
alpha = 0.01
weights = ones(n)   #initialize to all ones
for i in range(m):
    h = sigmoid(sum(dataMatrix[i]*weights)) #第i个子list，等效于matrix的第i行
    error = classLabels[i] - h
    weights = weights + alpha * error * dataMatrix[i]


# 改进版的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 步长曲线为先大后小，再大而后小，以此波浪式往复，每次波峰比上次低
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  #随机用某一行数据计算w，消除周期性
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])   #用完后，删除这行的行标，避免重复使用改行数据
    return weights
	


#西瓜书程序 watermelon
def loadDataSet_w():
    dataMat = []; labelMat = []
    fr = open('watermelon_1.txt')   #原始数据testSet的数据格式为：数据1，数据2，label
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #其中的1.0为迭代的初始值
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
    
dataArr_w, labelMat_w = loadDataSet_w()
weights_w = gradAscent(dataArr_w, labelMat_w)



def plotBestFit_w(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet_w()
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
    x = arange(0.1, 0.9, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    
plotBestFit_w(weights_w.getA())
'''