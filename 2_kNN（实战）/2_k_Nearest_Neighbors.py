# -*- coding: utf-8 -*-
import kNN
from numpy import *
import operator
from os import listdir

group, labels = kNN.createDataSet() #group type is array, labels type is list

'''
group
Out[9]: 
array([[ 1. ,  1.1],
       [ 1. ,  1. ],
       [ 0. ,  0. ],
       [ 0. ,  0.1]])

labels
Out[10]: ['A', 'A', 'B', 'B']
'''

kNN.classify0([0,0],group,labels,3)  #kNN算法的核心，给出矩阵数据，再给一个list（每行元素的类型），然后给出目标点的数据list，求出与之最近点，并用以命名其类型

'''
inX = [0,0]
dataSet = group
labels
k=3

dataSetSize = dataSet.shape[0] #4L
diffMat = tile(inX, (dataSetSize,1)) - dataSet #求出间距, 
sqDiffMat = diffMat**2  #间距的平方
sqDistances = sqDiffMat.sum(axis=1) #平方和
distances = sqDistances**0.5 #间距
sortedDistIndicies = distances.argsort() #从小到大的值对应的排名
classCount={}
  
for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]] #获取距离目标点最近的点的前k个点的序列
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #统计前k个点中，类型为'A'和'B'的数量分别为多少
sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)  ##operator, 使用元组的第二个元素进行排序, reverse=True -> 降序
sortedClassCount[0][0]
'''

datingDataMat, datingLabels = kNN.file2matrix('data/Ch02/datingTestSet2.txt') #将文本转化为数据矩阵和类型list
datingDataMat  #数据矩阵
datingLabels   #类型list
'''
filename = 'data/Ch02/datingTestSet2.txt'

fr = open(filename) #type is fr
numberOfLines = len(fr.readlines()) #get the number of lines in the file   ???
returnMat = zeros((numberOfLines,3))#prepare matrix to return, type of returnMat is array
classLabelVector = []   #prepare labels return   
fr = open(filename)
index = 0
for line in fr.readlines():  
    line = line.strip()       #type of line is str, 去除首末的空格字符
    listFromLine = line.split('\t')  #按照'\t'将str分割为list
    returnMat[index,:] = listFromLine[0:3]  #第index行的内容为listFromLine[0:3]，如果不用index，那么变为每一行元素内容都为该list
    classLabelVector.append(int(listFromLine[-1])) #list 元素为单个数字，故直接用list.append()最简单
    index += 1
'''

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy.random

font = FontProperties(fname='C:\Windows\Fonts\simfang.ttf',size=14)

fig = plt.figure()
ax = fig.add_subplot(111) #参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块

#################画出数据矩阵中所有的点####################

ax.scatter(datingDataMat[:,1], datingDataMat[:,2]) #每一行，第2列和第3列的数据，datingDataMat[:,2] 类型为 array，1000行，每行元素为一个数字

# datingDataMat[:,0]为获得的飞行常客里程数；datingDataMat[:,1]为'玩视频游戏所耗时间百分比'；datingDataMat[:,2]为'每周所消费的冰淇淋公升数'

plt.xlabel(u'玩视频游戏所耗时间百分比',fontproperties=font)
plt.ylabel(u'每周所消费的冰淇淋公升数',fontproperties=font)



#################画出数据矩阵中所有的点，并用颜色按照类型list进行分类####################

#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 150*array(random.randn(len(datingLabels))),15*array(datingLabels)) 
#其中按照datingLabels分类，150*array(random.randn(len(datingLabels)))描述不同类的点中心彩色部分的大小，15*array(datingLabels)描述点的颜色
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15*array(datingLabels),150*array(random.randn(len(datingLabels)))) 

ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15*array(datingLabels),15*array(datingLabels))


ax2 = fig.add_subplot(111) #参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax2.scatter(datingDataMat[:,0], datingDataMat[:,1],15*array(datingLabels),15*array(datingLabels)) 

#####带图例的图像####

f2 = plt.figure(2)  
idx_1=[]
idx_2=[]
idx_3 = []
i=0
while(i<len(datingLabels)):
    if datingLabels[i]==1:
        idx_1.append(i)
    if datingLabels[i]==2:
        idx_2.append(i)
    if datingLabels[i]==3:
        idx_3.append(i)
    i=i+1

f2 = plt.figure(2)  
p1 = plt.scatter(datingDataMat[idx_1,0], datingDataMat[idx_1,1], marker = 'x', color = 'm', label=u'不喜欢', s = 30)  
p2 = plt.scatter(datingDataMat[idx_2,0], datingDataMat[idx_2,1], marker = '+', color = 'c', label=u'魅力一般', s = 50)  
p3 = plt.scatter(datingDataMat[idx_3,0], datingDataMat[idx_3,1], marker = 'o', color = 'r', label=u'极具魅力', s = 15)  
plt.legend(prop=font, loc = 'upper left')  #prop = font显示出中文字，否则会显示出乱码


########对数据矩阵中的每列，进行归一化 newValue = (oldValue-min)/(max-min)#######

normMat, ranges, minVals=kNN.autoNorm(datingDataMat)
normMat
ranges
minVals

'''
dataSet = datingDataMat

minVals = dataSet.min(0) #沿着axis=0的方向取最小值，minVals type is list
maxVals = dataSet.max(0)
ranges = maxVals - minVals
normDataSet = zeros(shape(dataSet))
m = dataSet.shape[0] #沿着axis=0的方向，shape值是多少，本例中即行数
normDataSet = dataSet - tile(minVals, (m,1)) #堆叠，minVals(list)按行，将整个矩阵重复m次；按列，重复1次
normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
'''

######测试算法：用剩余的10%数据测算，之前建立的模型，算出某点的类型，是否与实际情况相吻合###

#kNN.datingClassTest()

'''
hoRatio = 0.10  #hold out 10%
datingDataMat, datingLabels = kNN.file2matrix('data/Ch02/datingTestSet2.txt') # datingDataMat: array; datingLabels: list
normMat, ranges, minVals = kNN.autoNorm(datingDataMat) #normMat: array; ranges: array; minVals: array
m = normMat.shape[0] #m: long，数据矩阵总行数
numTestVecs = int(m*hoRatio)
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = kNN.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],10) #classifierResult: int;前100个用来检验，后900个用来建立模型
    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
    if (classifierResult != datingLabels[i]): errorCount += 1.0  #统计错误总数
print "the total error rate is: %f" % (errorCount/float(numTestVecs)) #统计概率
print errorCount
'''
####### 用raw_input 输入某种情况，用模型算出结果，给出结论 ######

kNN.classifyPerson()

############ 将 32x32的矩阵，转化为1x1024的矩阵 ##############

testVector = kNN.img2vector('data/Ch02/digits/testDigits/0_13.txt')

testVector[0,0:31]
testVector[0,32:63]


######## 识别手写数字 #########
#先将训练集中的数字矩阵，由32x32，转化为1x1024；
#再导入进array，用每个训练文件建立一行，m个训练文件，建立一列，建立数据矩阵 m*1024
#从每个训练文件的文件名中提取label，建立label的list
#再用几个检验矩阵，带进模型中计算，验证下准确率

kNN.handwritingClassTest()


