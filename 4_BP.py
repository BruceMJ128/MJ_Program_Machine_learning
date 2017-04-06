# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import pandas as pd

def loadDataSet():
    frame = pd.read_csv('watermelon_3.0_84_data.csv')
    dataMat = mat(frame.ix[:,0:8].as_matrix())
    labelMat = mat(frame.ix[:,8:10].as_matrix())
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def cumulated_BP(eta, Ek_iterationtime,q):  #eta: η 迭代步长(又成为学习率)；Ek_iterationtime: 每个Ek进行迭代的最大次数; q: 隐层的宽度，本例中q = d+1
    dataMat, labelMat = loadDataSet()
    m, d = shape(dataMat)
    m, l = shape(labelMat)
    
    #初始化阀值和连接权    
    gamma = (mat(np.random.randn(q))).transpose()    #阀值gamma: γ
    theta = (mat(np.random.randn(l))).transpose()    #阀值thema: θ
    v = mat(np.random.randn(d,q))                  #输入层与隐层之间的连接权：ν_ih
    w = mat(np.random.randn(q,l))                  #隐层与输出层之间的连接权：w_hj
    
    #迭代
    t = 0                 #迭代次数
    E_total =float(1.0)   #误差累积总和
    while((E_total>0.01) and (t<Ek_iterationtime)):
        #更新迭代次数
        t = t+1         
        
        #每次迭代前，各中间变量要归0，尤其是E和E_total
        E = (mat(zeros(m))).transpose()
        E_total = 0   #总误差
        b = mat(zeros((m,q)))    #b: 隐层输出    
        y = mat(zeros((m,l)))    #输出拟合值：y        
        
        #计算本次迭代的E_total:
        alpha = dataMat * v     #隐层的输入：Alpha (α)
        for k in range(m):
            for h in range(q):
                b[k,h]=sigmoid(alpha[k,h]-gamma[h])        
        beta =b*w             #隐层的输出：b
        for k in range(m):
            for j in range(l):
                y[k,j]=sigmoid(beta[k,j]-theta[j])
                E[k]=float(E[k])+0.5*(y[k,j]-labelMat[k,j])**2
        for k in range(m):
                E_total = E_total+float(1.0)/float(m)*float(E[k])
        
        print 't: ',t,' E_total: ',E_total,'\n'
        print 'theta: ',theta, '\n'                
        #计算下一步迭代的各参数：w,v,theta,gamma
        g = mat(zeros((m,l)))               #中间变量：g_j
        e = mat(zeros((m,q)))               #中间变量：e_h        
        for k in range(m):
            for j in range(l):
                g[k,j]=y[k,j]*(1-y[k,j])*(labelMat[k,j]-y[k,j])
        for k in range(m):
            for h in range(q):
                e[k,h]=b[k,h]*(1-b[k,h])*(w[h]*g[k].transpose())
                
        w = w + eta*b.transpose()*g
        v = v + eta*dataMat.transpose()*e        
        for j in range(l):
            theta[j]=theta[j]-eta*average(g.transpose()[j])        #此处将Δθ_j = -η*g_j，改为了：Δθ_j = -(1/m)∑ η*g_kj
        for h in range(q):
            gamma[h]=gamma[h]-eta*average(e.transpose()[h])
            
    return E_total,y

dataMat, labelMat = loadDataSet()
m, d = shape(dataMat)
m, l = shape(labelMat)
q = d+1
eta=1
Ek_iterationtime = 200
E_total,y = cumulated_BP(eta, Ek_iterationtime,q)
    
def standard_BP(eta, Ek_iterationtime):  #eta: η 迭代步长；Ek_iterationtime: 每个Ek进行迭代的最大次数
    return 0