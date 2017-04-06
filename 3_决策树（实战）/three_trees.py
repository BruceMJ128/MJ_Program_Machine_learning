# -*- coding: utf-8 -*-
from math import log
import operator

#定义函数计算香农信息熵（ID3算法）
def calcShannonEnt(dataSet):   #dataSet类型为 n*m 的二维list
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  #用子list featVec的最后一位作为label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1 #统计label的种类和数量
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -=prob * log(prob, 2)   #计算的是在按照要素划分之前，label本身的香农熵是多少
    return shannonEnt

#给定一组数据
#dataSet是原始数据集，labels中的名称对应的是dataSet中轴的名称：'no surfacing':0,1; 'flippers':10,11
def createDataSet():
    dataSet = [[1,11,'yes'],[1,11,'yes'],[1,10,'no'],[0,11,'no'],[0,11,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

myDat, labels = createDataSet()
ShEnt_origin = calcShannonEnt(myDat) #计算的是在按照要素划分之前，label本身的香农熵是多少
#-(2/5*log2(2/5)+3/5*log2(3/5))

#将二维list dataSet，按照轴axis划分，提取值为value的部分,提取后的内容仍为二维list
def splitDataSet(dataSet, axis, value):  
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #提取除了axis轴以外所有信息
            reducedFeatVec.extend(featVec[axis+1:]) #注意extend和append的用法
            retDataSet.append(reducedFeatVec)
    return retDataSet

myDat, lables = createDataSet()
List_0_1 = splitDataSet(myDat, 0,1)
List_0_0 = splitDataSet(myDat, 0,0)

#选择最好的数据划分方式，即选择划分后香农熵减小最大的特征值
#选择出香农熵增最高的特征值，bestFeat为特征值对应的轴
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1 #特征值的种类数
    baseEntropy = calcShannonEnt(dataSet) #计算初始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #按列整理数据：将第i列整理到一个list中
        uniqueVals = set(featList) #列出第i列中的所有元素
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value) #将初始二维list按照第i列，值为value划分，得到一个子list
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)#计算划分后的香农熵,并乘以相应的比例
        infoGain = baseEntropy - newEntropy #与初始状态相比，按照第i列划分，计算熵增
        if (infoGain > bestInfoGain):  #找寻最大的熵增结果，并记录最大熵增对应的特征值
            bestInfoGain = infoGain  
            bestFeature = i
    return bestFeature

myDat, labels = createDataSet()
BF = chooseBestFeatureToSplit(myDat)



#递归构建决策树
# 输入List，返回List中频次最高的值，在决策树中，该值作为叶节点的代表值
def majorityCnt(classList): 
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] =0
        classCount[vote] += 1  #统计classCount中各元素出现的频次
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    #将classCount按照第二个领域进行排序，排序方法为降序
    return sortedClassCount[0][0]  #返回 classList中出现频次最高的值 

myDat, labels = createDataSet()
LastCol = [example[-1] for example in myDat]
x = majorityCnt(LastCol)

#dataSet为多维list，反应数据源 ；labels为一维list，反应每列的影响因子的名称
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  #将label列组成list

    #递归的两种结束方式
    if classList.count(classList[0]) == len(classList): # 当label List中只有一种值 stop splitting when all of the classes are equal
        return classList[0]    
    if len(dataSet[0]) == 1:     #当dataSet中只剩余一行，无法继续划分 stop splitting when there are no more features in dataSet 
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)    #选择出香农熵增最高的特征值，bestFeat为特征值对应的轴
    bestFeatLabel = labels[bestFeat]    #香农熵增最高的特征值，对应的名称
    myTree = {bestFeatLabel:{}}      
    del(labels[bestFeat])     #删除labels中上一层中的最优熵增特征值的名称
    featValues = [example[bestFeat] for example in dataSet] #最优熵增特征值的列转化为list
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #复制labels，用该复制值进行运算，以免改变labels，此时labels中上一级最优熵的特征值已经被删除。copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) 
        #新建二维数组myTree并赋值，采用递归的方法
    return myTree

myDat, labels = createDataSet()
myTree = createTree(myDat, lables)

#使用决策树的分类函数
#输入变量：决策树，x轴的labels名称，验证list
#输入验证list，按照原先建立的决策树模型，进行推演，得到结果如下：叶子为深色 1，西瓜圆形1，西瓜是否甜的结果为是'yes'，即testVec=[1,1], classLabel = 'yes'
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #得到firstStr这个labels对应的列的排序
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]    
    if isinstance(valueOfFeat, dict):   #isinstance函数，判断valyeOfFeat的数据类型是不是dict，如果是返回tree，否则返回False
        classLabel = classify(valueOfFeat, featLabels, testVec)  #递归
    else: classLabel = valueOfFeat
    return classLabel
    
myDat, labels = createDataSet()

import three_treePlotter

myTree = three_treePlotter.retrieveTree(0) #决策树

x = classify(myTree, labels, [1,0])
y = classify(myTree, labels, [1,1])


#用模块pickle序列化决策树，并存储在txt文件中
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

#调用储存在txt中的pickle模块化信息，并转化为dict
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

storeTree(myTree, 'classifierStorage.txt')
grabTree('classifierStorage.txt')

#示例：使用决策树预测隐形眼镜类型
fr = open('lenses.txt')  
lenses = [inst.strip().split('\t') for inst in fr.readlines()]  #将文件中的数据转化为多维list，用于建立模型
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses, lensesLabels)

print lensesTree

three_treePlotter.createPlot(lensesTree)

