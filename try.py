# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import pandas as pd
import math
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten

arr_x= array([[0,0],[0,1],[1,0],[1,1]])

m,d=shape(arr_x)

vec_y =array([0,1,1,0])
eta = 0.125
vec_Phi = zeros(4)

q=5
arr_P = zeros((m,q))
vec_Phi = zeros(m)
vec_w = np.random.uniform(0,1,size=q)
vec_beta = np.random.uniform(0,1,size=q)
#vec_w = array([0.1, 0.2, 0.3, 0.4, 0.5])
#vec_beta = array([0.9, 0.8, 0.7, 0.6, 0.5])

eta = 0.125
arr_c = np.random.uniform(0,1,size=(q,d))
#arr_c = array([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6],[0.7, 0.8],[0.9, 0.05]])



t = 0                 #迭代次数
E =float(1.0)        #误差累积总和
old_E = float(0.0)   #上一次迭代的累积误差
f = 0.0             #frequency: 统计迭代误差之差小于0.0001的次数

for time in range(2000):
    for k in range(m):
        for i in range(q):                
            arr_P[k,i]=math.exp(-vec_beta[i]*(linalg.norm(arr_x[k]-arr_c[i]))**2)
        vec_Phi[k]=(mat(vec_w)*mat(arr_P[k]).transpose()).getA()
    
    E=((mat(vec_Phi)-mat(vec_y))*((mat(vec_Phi)-mat(vec_y)).transpose())).getA()[0][0]
    
    dE_dw = zeros(q)
    dE_dbeta = zeros(q)                
    for i in range(q):            
                for k in range(m):                 
                    dE_dw[i] = dE_dw[i]+(mat(vec_Phi[k])-mat(vec_y[k]))*mat(arr_P[k,i])
                    dE_dbeta[i] = dE_dbeta[i] - (vec_Phi[k]-vec_y[k])*vec_w[i]*arr_P[k,i]*(linalg.norm(arr_x[k]-arr_c[i]))**2
            
    vec_w = vec_w - eta*dE_dw
    vec_beta=vec_beta-eta*dE_dbeta
    print E