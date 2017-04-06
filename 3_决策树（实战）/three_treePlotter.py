# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#绘制树节点，包括箭头，箭头目标处的内容和方框
#annotations: 注解工具

#示例： plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
# plotNode(箭头终止处label，箭头终点，箭头起始点，箭头终点label方框信息)
def plotNode(nodeTxt, centerPt, parentPt, nodeType):  #数据类型：str, 二维元组tuple，元组，dict
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
             
#xycoords: 反应箭头终点坐标的类型
#bbox: 类型为dict，包括boxstyle方框种类 和 fc 方框中灰色的深浅程度facecolor（0.2为深，0.8为浅）
#arrowprops: 类型为dict，包括箭头种类 arrowstyle

#old edition
                                   
def createPlot_old():
    fig = plt.figure(1, facecolor='white')
    fig.clf()  #清除当前图像窗口
    createPlot_old.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

#formal edition: 根据dict内容，画出树状图见下方createPlot(inTree)



#获取叶节点的数目，以解决如何放置所有的树节点
#myTree: {'no surfacing': {0: 'no', 1: {'flippers': {10: 'no', 11: 'yes'}}}}
#写法好牛，注意学习
def getNumLeafs(myTree):  #返回值为int
    numLeafs = 0
    firstStr = myTree.keys()[0]     #type is str or int
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':  #test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])  #递归，可以算出各个分支的节点数
        else:   numLeafs +=1
    return numLeafs

#获取树的层数
#想法好牛
#这里得到的并不是数据的层数 c,而是箭头的高度 h, c =h+1，用3_trees中的myTree,数据有三层，但高度只有2
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

#正式createPlot版本

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))  #树的叶节点数目
    plotTree.totalD = float(getTreeDepth(inTree)) #树的层高
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;  #plotTree初始 xOff 和 yOff，用于待遇plotTree运算
    plotTree(inTree, (0.5,1.0), '')
    plt.show()



#计算宽与高
def plotTree(myTree, parentPt, nodeTxt):#parentPt：起始点位置。if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree 树的宽度小于叶子的总数
    depth = getTreeDepth(myTree)    #树的高度
    firstStr = myTree.keys()[0]     #节点的label。the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff) #箭头终点，对于最高点cntrPt(0.5,1.0)，该点与起始点位置相同，故看不到箭头，但可以显示label
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode) # plotNode(箭头终止处label，箭头终点，箭头起始点，箭头终点label方框信息)，此时默认了第一个箭头终点label类型为decisionNode，因为第一个点在原地打转，肯定是树枝节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #第二层的高度
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion 递归            
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  #第一次循环，xOff仍为-0.125,计算公式：-0.125+1/4
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #为什么此处要将yOff复位到上一层的高度，为了使同一级别的点在运算中，高度保持一致
#if you do get a dictonary you know it's a tree, and the first element will be another dict

#在父子节点之间填充文本信息，此例中，将dict的key作为label，添加在箭头的正中
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
myTree = retrieveTree(0)
createPlot(myTree)

myTree['no surfacing'][3]='maybe'
createPlot(myTree)