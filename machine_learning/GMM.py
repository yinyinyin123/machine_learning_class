# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 15:03:29 2017

@author: dell
"""

import numpy as np
from numpy.linalg import cholesky
import math
from sklearn.datasets import load_iris

K = 4
length = 0

#样本属于类别i的概率  
def p_x_lamda(data,m,sig):  
    #print data,m,sig
    data = np.array(data)
    data = data - m
    t =  np.linalg.det(sig)
    temp =  np.dot(data,np.linalg.inv(sig))
    return  math.exp((-1.0/2)*np.dot(temp,data))*(1.0/(math.sqrt(t)))
#log极大似然值
def mle(train_data,mu,sigma,p):
    log_likehood = 0.0
    for i in range(0,len(train_data)):
        temp = 0.0
        for j in range(0,K):
            temp = temp + p_x_lamda(train_data[i],mu[j],sigma[j])*(1.0/(math.sqrt(math.pow(2*math.pi,D))))*p[j]
        log_likehood = log_likehood + math.log(temp)
    print log_likehood
    
    
def EM(train_data):
    N = len(train_data[0])
    time = 0
    #对mu，sigma,pi的初始设置
    mu = []
    sigma = []
    p = [1.0/K for i in range(0,N)]
    mu = np.random.rand(K,N)
    for i in range(0,K):
        sigma.append(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    sigma = np.array(sigma)
    #迭代 
    while(time < 20):
        #极大似然值
        mle(train_data,mu,sigma,p)
        time = time + 1
        #  E - step，计算样本属于各个类别的i
        E = []
        for i in range(0,length):
            temp = []
            for j in range(0,K):  #N->k
                #print j,
                temp.append(p[j] * p_x_lamda(train_data[i],mu[j],sigma[j]))
            #归一化
            a = sum(temp)
            for k in range(0,len(temp)):
                temp[k] = temp[k] / a
            E.append(temp)
        # M -step
        #重新计算mu，pi
        for i in range(0,K):  #N->k
            fenzi = [0.0 for m in range(0,N)]
            fenmu = 0.0
            for j in range(0,length):
                fenmu = fenmu + E[j][i]
                fenzi = fenzi + E[j][i] * train_data[j]
            mu[i] = fenzi / fenmu
            p[i] = fenmu / length
        #重新计算sigma
        for i in range(0,K):  #N->k
            fenzi = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
            fenmu = 0.0
            for j in range(0,length):
                fenmu = fenmu + E[j][i]
                a = np.array([train_data[j]])
                a = a - mu[i]
                fenzi = fenzi + np.dot(a.T,a) * E[j][i]
            #print fenzi
            sigma[i] = fenzi / fenmu
            
    #迭代结束之后，计算分裂结果   
    E = []
    print "out"
    for i in range(0,length):
        temp = []
        for j in range(0,K):  #N->k
                #print j,
            temp.append(p[j] * p_x_lamda(train_data[i],mu[j],sigma[j]))
        a = sum(temp)
        for k in range(0,len(temp)):
            temp[k] = temp[k] / a
        E.append(temp)                    
    print "mu:"
    for i in range(0,K):#N->k
        print mu[i]
    print "分类情况"
    for i in range(0,4):
        coun = [0 for k in range(0,K)]
        for j in range(i*500,(i+1)*500):
            p = 0
            te = 0
            for m in range(0,N):
                if(E[j][m] > te):
                    p = m
                    te = E[j][m]
            coun[p] = coun[p] + 1
        print coun
    """
    for i in range(0,3):
        coun = [0 for k in range(0,K)]
        for j in range(i*50,(i+1)*50):
            p = 0
            te = 0
            for m in range(0,K):#N->k
                if(E[j][m] > te):
                    p = m
                    te = E[j][m]
            coun[p] = coun[p] + 1
        print coun
    """
    

#设置高斯分布的sigma
sigma1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
sigma2=np.array([[0.81,0,0,0],[0,0.81,0,0],[0,0,0.81,0],[0,0,0,0.81]])
sigma3=np.array([[0.25,0,0,0],[0,0.25,0,0],[0,0,0.25,0],[0,0,0,0.25]])
sigma4=np.array([[1,0,0,0],[0,0.81,0,0],[0,0,0.81,0],[0,0,0,0.81]])

R1= cholesky(sigma1)
R2= cholesky(sigma2)
R3= cholesky(sigma3)
R4= cholesky(sigma4)
#设置高斯分布的mu
mu_1 = np.array([[4,4,4,20]])
mu_2 = np.array([[14,6,6,6]])
mu_3 = np.array([[8,1,8,8]])
mu_4 = np.array([[1,15,15,4]])
#生成4维高斯分布数据，
s_1 = np.dot(np.random.randn(500,4),R1)+mu_1
s_2 = np.dot(np.random.randn(500,4),R2)+mu_2
s_3 = np.dot(np.random.randn(500,4),R3)+mu_3
s_4 = np.dot(np.random.randn(500,4),R4)+mu_4


#将数组拼接 形成训练数据 
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_3:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_4:
    list_temp = list(data)
    list_1.append(list_temp)
train_data = np.array(list_1)
D = len(train_data[0])
length = len(train_data)
#print train_data
"""

data = load_iris()
features = data['data']
labels=data['target_names'][data['target']]
features_names=data['feature_names']
target_names=data['target_names']
target=data['target']
plength = features[:,2]
is_setosa = (labels == 'setosa')


length = len(features)
train_data = np.array(features)
"""

EM(train_data)




