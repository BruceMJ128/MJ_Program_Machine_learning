# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import pandas as pd
import math

def RBF(eta,arr_x,vec_y): #eta(η):学习率（步长）,x为m*d二维矩阵，y为长度为m的list
    m,d=shape(arr_x)
    q = 10
    
    #隐层为q*d维矩阵，将arr_x映射到更高维度空间，使得原本不可拆分的m*d维数据线性可分，前提条件：q>m    
    #随机化算法初始化神经元中心 arr_c: 效果不好，拟合值迭代后都趋向1
    arr_P = zeros((m,q))   #ρ(xk,ci)
    vec_Phi = zeros(m)     #拟合值φ(x)
    vec_w = np.random.uniform(0,1,size=q)
    vec_beta = np.random.uniform(0,1,size=q)
    arr_c = np.random.uniform(0,1,size=(q,d))
    
    savetxt("c.txt", arr_c)                  #numpy.savetxt("c.txt", arr_c): 将随机产生的c向量保存到文件 c.txt
      
        
    #迭代vec_w和vec_beta，目标使E最小化
    t = 0                 #迭代次数
    E =float(1.0)        #误差累积总和
    old_E = float(0.0)   #上一次迭代的累积误差
    f = 0.0             #frequency: 统计迭代误差之差小于0.0001的次数
    
    while(t<500):
        t = t+1
        #计算中间过渡矩阵 P
        for k in range(m):
            for i in range(q):                        
                arr_P[k,i]=math.exp(-vec_beta[i]*(linalg.norm(arr_x[k]-arr_c[i]))**2)
            vec_Phi[k]=(mat(vec_w)*mat(arr_P[k]).transpose()).getA()
               
        #计算累积误差
        E=0.5*((mat(vec_Phi)-mat(vec_y))*((mat(vec_Phi)-mat(vec_y)).transpose())).getA()[0][0]
        
        #决策迭代是否继续进行
        if(abs(old_E-E)<0.0001):
            f=f+1
            if(f>10):    #当连续10次迭代误差之差小于0.0001，迭代停止，模型稳定，已达到拟合的极限
                break
        else:
            old_E=E
            f=0.0        
        
        #迭代w 和 β
        #dE_dw = ∑1~m [(φ(xk)-yk)*ρ(xk,ci)
        #dE_dbeta = -∑1~m [(φ(xk)-yk)*wi*ρ(xk,ci)*||vec_xk-vec_ci||^2]        
        dE_dw = zeros(q)
        dE_dbeta = zeros(q)                
        for k in range(m):
            for i in range(q):                                         
                dE_dw[i] = dE_dw[i]+(mat(vec_Phi[k])-mat(vec_y[k]))*mat(arr_P[k,i])
                dE_dbeta[i] = dE_dbeta[i] - (vec_Phi[k]-vec_y[k])*vec_w[i]*arr_P[k,i]*(linalg.norm(arr_x[k]-arr_c[i]))**2
        
        vec_w = vec_w - eta*dE_dw
        vec_beta=vec_beta-eta*dE_dbeta
        print 't:',t,'  E:',E,'\n'               
    return vec_Phi, vec_w, vec_beta, arr_c

arr_x= [[0,0],[0,1],[1,0],[1,1]]
vec_y =[0,1,1,0]
eta = 0.125
vec_Phi = zeros(4)

vec_Phi, vec_w, vec_beta, arr_c = RBF(eta,arr_x,vec_y)