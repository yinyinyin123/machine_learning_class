# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:03:23 2017

@author: dell
"""


import numpy as np
from numpy.linalg import cholesky
import math

#设置高斯分布的sigma
sigma=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
#sigma = np.array([[1,0.25,0.25,0.25],[0.25,1,0.25,0.25],[0.25,0.25,1,0.25],[0.25,0.25,0.25,1]])
R = cholesky(sigma)

#设置高斯分布的mu
mu_1 = np.array([[1,1,1,1]]);
mu_2 = np.array([[2,2,2,2]])

#生成4维高斯分布数据，两个类别
s_1 = np.dot(np.random.randn(50,4),R)+mu_1
s_2 = np.dot(np.random.randn(50,4),R)+mu_2
l_1 = np.zeros((1,50))
l_2 = np.ones((1,50))

#将数组拼接 形成训练数据 
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
list_2 = []
for data in l_1:
    for ele in data.flat:
        list_2.append(ele)
for data in l_2:
    for ele in data.flat:
        list_2.append(ele)
train_data = np.array(list_1)
data_label = np.array(list_2)
#print train_data
#print data_label
#生成4维高斯分布数据，两个类别
s_1 = np.dot(np.random.randn(1000,4),R)+mu_1
s_2 = np.dot(np.random.randn(1000,4),R)+mu_2
l_1 = np.zeros((1,1000))
l_2 = np.ones((1,1000))

#将数组拼接 形成测试数据
list_1 = []
for data in s_1:
    list_temp = list(data)
    list_1.append(list_temp)
for data in s_2:
    list_temp = list(data)
    list_1.append(list_temp)
list_2 = []
for data in l_1:
    for ele in data.flat:
        list_2.append(ele)
for data in l_2:
    for ele in data.flat:
        list_2.append(ele)
test_data = np.array(list_1)
test_label = np.array(list_2)

#计算WT x
def calculate_wtx(data,w,w0):
    sum = w0*1 + 0.0
    for i in range(0,len(data)):
        sum = sum + data[i]*w[i]
    return sum
def calculate_wtxx(data,w):
    sum = 0.0
    for i in range(0,len(w)):
        sum = sum + data[i]*w[i]
    return sum
        
#计算p(y = 1 | x,w) 
def calculate_py1(data,w,w0):
    wtx = calculate_wtx(data,w,w0)
    return 1-1.0/(1+math.exp(wtx))
def calculate_pyy1(data,w):
    wtx = calculate_wtxx(data,w)
    return 1-1.0/(1+math.exp(wtx))
    
#计算正确率  
def calcul_correct(w0,w,test_data,test_label):
    count = 0.0
    for i in range(0,len(test_data)):
        py_1 = calculate_py1(test_data[i],w,w0)
        if(py_1 >= 0.5 and test_label[i] == 1):
            count = count + 1
        elif(py_1 < 0.5 and test_label[i] == 0):
            count = count + 1
    print "测试样本正确率",count/len(test_data)
def calcul_correctt(w,test_data,test_label):
    count = 0.0
    for i in range(0,len(test_data)):
        py_1 = calculate_pyy1(test_data[i],w)
        if(py_1 >= 0.5 and test_label[i] == 1):
            count = count + 1
        elif(py_1 < 0.5 and test_label[i] == 0):
            count = count + 1            
            #print "错误"
    print "测试样本正确率",count/len(test_data)
    
#计算迭代前后俩点的距离
def distance(w_temp,w):
    sum = 0.0
    for i in range(0,len(w_temp[0])):
        sum = sum + math.pow(w_temp[0][i]-w[i],2)
    return sum
                 
#训练模型 参数 并测试，梯度上升
def train_w_and_test_nopunish_graident(train_data,data_label,test_data,test_label,lamda):
    
    #自己生成的数据的w初始值
    w0 = 0
    w  = [0.5,0.5,0.5,0.5]
    #w=[0,0,0,0,0,0]
    #uci的数据的w初始值
    #w0 = 0.0
    #w=[0,0,0,0]
    l = -1e10
    while(True):
        print "----"
        w0_temp = w0
        w_temp = [1,1,1,1]
        for i in range(0,len(w)):
            w_temp[i] = w[i]
        for i  in range(0,len(train_data)):
            py_1 = calculate_py1(train_data[i],w_temp,w0_temp)
            #对自己生成的数据，参数调整 
            w0 = w0 + 0.00001*(data_label[i] - py_1)
            #w0 = w0 + 0.0000005*(data_label[i] - py_1)
            #对uci的数据，参数调整
            #w0 = w0 + 0.000000003*(data_label[i] - py_1)
        w0 = w0 - lamda*w0_temp
        for i in range(0,len(w)):
            for j in range(0,len(train_data)):
                py_1 = calculate_py1(train_data[j],w_temp,w0_temp)
                #自己生成的数据，参数调整
                w[i] = w[i] + 0.00001*(train_data[j][i] * (data_label[j] - py_1)) 
                #对uci的数据，参数调整
               # w[i] = w[i] + 0.0000005*(train_data[j][i] * (data_label[j] - py_1))
                #w[i] = w[i] + 0.000000003*(train_data[j][i] * (data_label[j] - py_1)) 
        #print w0,w
        for i in range(0,len(w)):
            w[i] = w[i] - w_temp[i]*lamda
        sum = 0
        for i in range(0,len(train_data)):
            wtx = calculate_wtx(train_data[i],w,w0)
            sum = sum + data_label[i]*wtx - math.log(1+math.exp(wtx),math.e)
        correct_p = math.pow(math.exp(sum),1.0/len(train_data))
        print "训练数据拟合率",correct_p
        #correct_p = math.pow(math.exp(sum),1.0/600)
        #print "训练数据拟合率",correct_p
        #真是数据为0.000001
        if(correct_p - l < 0.000001 ):
            calcul_correct(w0,w,test_data,test_label)
            print w0,w
            break;
        else:
            l = correct_p
            
##训练模型 参数 并测试，牛顿
def train_w_and_test_nopunish_nuton(train_data,data_label,test_data,test_label,lamda):
     #自己生成的数据,w初始值     
     #w = [0.1,0.1,0.1,0.1,0.1]
     #uci生成的数据，w初始值
     w = [0,0,0,0,0,0,0]
     count = 0
     dis = 1000
     while(True):
         count = count + 1 
         H =  np.zeros((len(w),len(w)))
         for i in range(0,len(H)):
             for j in range(0,len(H[i])):
                 for k in range(0,len(train_data)):
                     wtx = calculate_wtxx(train_data[k],w)
                     #print 'count:',count,'k:',k,'wtx:',wtx
                     H[i][j] = H[i][j] + train_data[k][i]*train_data[k][j]*math.pow((1+math.exp(wtx)),-2)*(-1)*math.exp(wtx) - (1.0)*lamda/len(train_data)
         
         partial = np.zeros((1,len(w)))
         for i in range(0,len(partial[0])):
             for j in range(0,len(train_data)):
                 py_1 = calculate_pyy1(train_data[j],w)
                 partial[0][i] = partial[0][i] + train_data[j][i]*(data_label[j] - py_1)
         for i in range(0,len(partial[0])):
             partial[0][i] = partial[0][i] - lamda*partial[0][i]
         w_temp = np.array(w)
         H_1_partial = np.dot(np.linalg.inv(H),partial.T)
         w_temp = w_temp - H_1_partial.T
         dis_new = distance(w_temp,w)
         if(abs(dis-dis_new) <1e-5):
              break
         else:
             dis = dis_new
             w = list(w_temp[0])
     sum = 0
     for i in range(0,len(train_data)):
         wtx = calculate_wtxx(train_data[i],w)
         sum = sum + data_label[i]*wtx - math.log(1+math.exp(wtx),math.e)
     print "训练样本拟合率:",math.pow(math.exp(sum),1.0/len(train_data))
     print "迭代次数:",count
     calcul_correctt(w,test_data,test_label)
         
                 

print "------------梯度下降法 无惩罚项------------------" 
lamda = 0
print "$$$$$$$$"
train_w_and_test_nopunish_graident(train_data,data_label,test_data,test_label,lamda)
print "------------------------------------------------"
print "------------牛顿法 无惩罚项"
#将数据变换，为了将w0也能加入w中，所以在每个训练数据和测试数据前都加一个

list_1 = []
for data in train_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_1.append(list_temp)
train_data = np.array(list_1)
list_2 = []
for data in test_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_2.append(list_temp)    
test_data = np.array(list_2) 
#train_w_and_test_nopunish_nuton(train_data,data_label,test_data,test_label,lamda)


train_data = []
data_label = []
test_data = []
test_label = []
f = open('test.txt','r')
dataset = f.readlines()
count = 0
for data in dataset:
    count = count + 1
    list_temp = []
    l = data.split(',')
    if(count < 200):
        for i in range(0,6):
            list_temp.append(float(l[i]))
        train_data.append(list_temp)
        data_label.append(int(l[6])-1)
    else:
        for i in range(0,6):
            list_temp.append(float(l[i]))
        test_data.append(list_temp)
        test_label.append(int(l[6])-1)

train_data = np.array(train_data)
data_label = np.array(data_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
#train_w_and_test_nopunish_graident(train_data,data_label,test_data,test_label,lamda)



list_1 = []
for data in train_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_1.append(list_temp)
train_data = np.array(list_1)
list_2 = []
for data in test_data:
    list_temp = list(data)
    list_temp.insert(0,1)
    list_2.append(list_temp)    
test_data = np.array(list_2)

#train_w_and_test_nopunish_nuton(train_data,data_label,test_data,test_label,lamda)


