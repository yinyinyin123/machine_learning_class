# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:48:37 2017

@author: dell
"""

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random
import math

from sklearn.datasets import load_iris
data = load_iris()
features = data['data']
labels=data['target_names'][data['target']]
features_names=data['feature_names']
target_names=data['target_names']
target=data['target']
nimei = data['target']
plength = features[:,2]
matrix_1 = np.array([features[1]])
is_setosa = (labels == 'setosa')


for t,marker,c in zip(xrange(3),">ox","rgb"):    
    plt.scatter(features[target == t,1],features[target==t,2],marker=marker,c=c)
    
s = []
while(len(s)<20):
    x = random.randint(0,149)
    if x not in s:
        s.append(x)
s.sort()
i = 0
j = 0
b=[]
lab=[]
for row in features:
    
    if(j<len(s) and i!=s[j] ):
        b.append(row)
        lab.append(labels[i])
    elif(j<len(s) and i==s[j]):
        j = j+1
    else:
        b.append(row)
        lab.append(labels[i])
    i = i+1
labels=np.array(lab)
f = np.array(b)
target=['setosa','versicolor','virginica']
a =  np.array([[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]])

e = np.array([[3,4,3,4]])
w = np.array([a[0]])



l = 1e-300
while(1>0):
    for i in range(0,len(f)):
        sum2 = 0.0
        matrix_1 = np.array([f[i]])
        tt = [0,0,0]
        for j in range(0,3):
            matrix_2 = np.array([a[j]])
            temp =  np.dot(matrix_2,matrix_1.T)
            sum2 = sum2 + math.exp(temp[0][0])
            tt[j]=math.exp(temp[0][0])
        
        if(labels[i] == 'setosa'):
            for j in range(0,4):
            
                a[0][j] = a[0][j] + f[i][j]*0.0001
                a[0][j] = a[0][j] - 0.0001*tt[0]*f[i][j]/sum2
        elif(labels[i]=='versicolor'):
            for j in range(0,4):
            
                a[1][j]=a[1][j] + 0.0001*f[i][j]
                a[1][j]=a[1][j] - 0.0001*tt[1]*f[i][j]/sum2       
        else:
        
            for j in range(0,4):
            
                a[2][j] = a[2][j] + 0.0001*f[i][j]
                a[2][j] = a[2][j] - 0.0001*tt[2]*f[i][j]/sum2
    l_new = 1
    sum_1 = 0.0
    for i in range(0,len(f)):
        matrix_1 = np.array([f[i]])
        sum_0 = 0.0
        #print matrix_1
        for j in range(0,3):
            matrix_2 = np.array([a[j]])
            temp = np.dot(matrix_2,matrix_1.T)
            sum_0 = sum_0 + math.exp(temp[0][0])*math.pow(10,-100)
            if(labels[i] == target[j]):
                sum_1 = math.exp(temp[0][0])*math.pow(10,-100)
                
        #print sum_1,sum_0
             # h = 1
        l_new = l_new * (sum_1/sum_0)
    if(l_new > l):
        l = l_new
    else:
        break
print "单个训练样本似然率:",math.pow(l,1.0/130)
count = 0.0
for i in range(0,len(s)):
    #print i
    k = 0
    p = 0
    matrix_1 = np.array([features[s[i]]])
    for j in range(0,3):
        matrix_2 = np.array([a[j]])
        temp = np.dot(matrix_2,matrix_1.T)
        if(temp[0][0] > p):
            p = temp[0][0]
            k = j
    if(k == nimei[s[i]]):
        count = count+1
print "测试样本正确率:",count/len(s)
    
            
        
        
        
    
        
            
            
       
                
        
            
            
        
                
                
                
    
            
        
        
    
    
    
    
























