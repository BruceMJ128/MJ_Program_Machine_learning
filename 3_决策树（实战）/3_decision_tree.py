# -*- coding: utf-8 -*-
import numpy as np
import trees
from math import log
import operator

myDat, labels = trees.createDataSet()
myDat

trees.calcShannonEnt(myDat)   #计算myDat这个list每个子列表最后一个元素的熵

'''
#def calcShannonEnt(dataSet): #计算香农熵
dataSet = myDat  #type is list [[x,y,z],[...],[],...]

numEntries = len(dataSet)
labelCounts = {}
for featVec in dataSet: #the the number of unique elements and their occurance
    currentLabel = featVec[-1]
    if currentLabel not in labelCounts.keys(): 
        labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1  #统计最后一位元素出现的次数
shannonEnt = 0.0
for key in labelCounts:
    prob = float(labelCounts[key])/numEntries #计算最后一位每个元素出现的概率
    shannonEnt -= prob * log(prob,2) #log base 2 #所有元素计算结果加和得到总的熵
#return shannonEnt
'''
    

myDat[0][-1]='maybe'
myDat

trees.splitDataSet(myDat,0,1)
trees.splitDataSet(myDat,0,0)

'''
#def splitDataSet(dataSet, axis, value):  #data type, dataSet: list; axis: int, value: int
#获得了所有包含value的子list，列出了第axis列，除了value以外，其他位置上的值，组成一个list作为返回值

dataSet = [[1, 'a', 'yes'],[1, 'b', 'yes'],[1, 'a', 'no'],[0, 'c', 'no'],[0, 'a', 'no']]
axis = 1
value = 'a'

retDataSet = []
for featVec in dataSet:
    if featVec[axis] == value: #list的单位list的第axis个元素值如果与目标值相等
        reducedFeatVec = featVec[:axis] #chop out axis used for splitting
        reducedFeatVec.extend(featVec[axis+1:])  #获得该list中除了目标元素value以外其他所有元素
        retDataSet.append(reducedFeatVec)

#返回值retDataSet: [[1, 'yes'], [1, 'no'], [0, 'no']] (返回所有第二个元素为'a'的list)
'''

dataSet = [[1, 'a', 'yes'],[0, 'c', 'no'],[1, 'b', 'yes'],[0, 'a', 'no'],[1, 'a', 'no']]
myDat, labels = trees.createDataSet()
myDat = dataSet
x = trees.chooseBestFeatureToSplit(myDat)

myDat

'''
#def chooseBestFeatureToSplit(dataSet):  #dataSet数据类型：list，返回值类型为int
#选择最好的数据划分方式，计算按照不同列划分，熵为多少；比较熵的大小，选择熵最小的方法，记录下熵，并记录下划分的列
numFeatures = len(dataSet[0]) - 1  #the last column is used for the labels，list的第一个子list的长度-1
baseEntropy = trees.calcShannonEnt(dataSet) #以最后一位作为label，计算初始状态香农熵
bestInfoGain = 0.0; bestFeature = -1
for i in range(numFeatures):#iterate over all the features
    featList = [example[i] for example in dataSet]    #create a list of all the examples of this feature
    #按照i=x的方式重新组合数列，如i=0,[1, 1, 1, 0, 0]
    uniqueVals = set(featList)   #get a set of unique values
    newEntropy = 0.0
    for value in uniqueVals:
        subDataSet = trees.splitDataSet(dataSet, i, value) #对于dataSet按照第i列检索，遇到值为value时，将其他位置的元素保存下来，最终存为一个总的list
        prob = len(subDataSet)/float(len(dataSet)) #计算总的dataSet中，第i列中，值为value所占总比
        newEntropy += prob * trees.calcShannonEnt(subDataSet) #整理数据以后，以最后一位作为label，计算子list的香农熵，分别乘以权重，再加和
    infoGain = baseEntropy - newEntropy #calculate the info gain; ie reduction in entropy，计算数据整理前后香农熵的减少值
    if (infoGain > bestInfoGain):   #compare this to the best gain so far
        bestInfoGain = infoGain #if better than current best, set to best
        bestFeature = i
        #因为数据重排只能选择某一列进行，比较按照各列重排，哪个熵最低，记录最低熵，并记录最低熵对应的列数

#    return bestFeature                      #returns an integer
'''
myTree = trees.createTree(myDat, labels)


#def majorityCnt(classList):
#返回次数最多的分类名称，返回类型为int

dataSet = [[1, 'a', 'yes'],[0, 'c', 'no'],[1, 'b', 'yes'],[0, 'a', 'no'],[1, 'a', 'no']]
classList = [example[-1] for example in dataSet]  #label list

classCount={}
for vote in classList:   #计数：classList中各元素出现的频次
    if vote not in classCount.keys(): classCount[vote] = 0
    classCount[vote] += 1
sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) ##使用元组的第二个元素进行排序
        
    #return sortedClassCount[0][0] #返回频次最高的元素名称
'''
#def createTree(dataSet,labels):  dataSet and labels type are list, 
#返回值为dataSet最后一列的某个元素

myDat, labels = trees.createDataSet()
dataSet = [[1, 'a', 'yes'],[0, 'c', 'no'],[1, 'b', 'yes'],[0, 'a', 'no'],[1, 'a', 'no']]

classList = [example[-1] for example in dataSet] #label list
if classList.count(classList[0]) == len(classList): 
    x= classList[0]#stop splitting when all of the classes are equal类别完全相同则停止继续划分
if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet，如果dataSet没有更多特征值，停止划分
    x= trees.majorityCnt(classList) #返回label中频次最高的元素名称，例如'no'
bestFeat = trees.chooseBestFeatureToSplit(dataSet)  #返回划分列的列数，可使香农熵最高，类型为int;遍历完所有特征时返回出现次数最多的,例如 0
bestFeatLabel = labels[bestFeat] #bestFeat划分列中的最优列，对应的label名称,例如'no surfacing'
myTree = {bestFeatLabel:{}}
del(labels[bestFeat]) #删除最优解，对应的labels中的列
featValues = [example[bestFeat] for example in dataSet] #按照香农熵最高的列划分，得到这列的元素
uniqueVals = set(featValues)
for value in uniqueVals:
    subLabels = labels[:]   #copy all of labels, so trees don't mess up existing labels
    myTree[bestFeatLabel][value] = trees.createTree(trees.splitDataSet(dataSet, bestFeat, value),subLabels)  #回到初始循环。再建构一次，嵌套结构,实现循环

x
myTree    
    #return myTree                           
'''

import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc=0.8)
arrow_args = dict(arrowstyle = "<-")

import treePlotter

#treePlotter.createPlot()
treePlotter.retrieveTree(1)

myTree = treePlotter.retrieveTree(0)

treePlotter.getNumLeafs(myTree)

treePlotter.getTreeDepth(myTree)

treePlotter.createPlot(myTree)

myTree['no surfacing'][3]='maybe'

treePlotter.createPlot(myTree)