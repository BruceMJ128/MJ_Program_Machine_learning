# -*- coding: utf-8 -*-
import kNN
from numpy import *
import operator
from os import listdir

######## 识别手写数字 #########
#先将训练集中的数字矩阵，由32x32，转化为1x1024；
#再导入进array，用每个训练文件建立一行，m个训练文件，建立一列，建立数据矩阵 m*1024
#从每个训练文件的文件名中提取label，建立label的list
#再用几个检验矩阵，带进模型中计算，验证下准确率

#kNN.handwritingClassTest()

'''
hwLabels = []  #作为label的list
trainingFileList = listdir('data/Ch02/digits/trainingDigits')   #load the training set, type of trainingFileList is list
m = len(trainingFileList)
trainingMat = zeros((m,1024)) #数据矩阵
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0] #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = kNN.img2vector('data/Ch02/digits/trainingDigits/%s' % fileNameStr) #将32*32的矩阵转化为1*1024，再写入大的矩阵中，作为一行
    
testFileList = listdir('data/Ch02/digits/testDigits')#iterate through the test set
errorCount = 0.0
mTest = len(testFileList)
for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0] #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = kNN.img2vector('data/Ch02/digits/testDigits/%s' % fileNameStr) #用做识别的样本
    classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
    if (classifierResult != classNumStr): errorCount += 1.0
print "\nthe total number of errors is: %d" % errorCount
print "\nthe total error rate is: %f" % (errorCount/float(mTest))
'''

kNN.handwriting()
