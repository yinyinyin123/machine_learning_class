# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
labels=data['target_names'][data['target']]
features_names=data['feature_names']
target_names=data['target_names']
target=data['target']
plength = features[:,2]
is_setosa = (labels == 'setosa')


#for t,marker,c in zip(xrange(3),">ox","rgb"):    
 #   plt.scatter(features[target == t,1],features[target==t,2],marker=marker,c=c)
    


    
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
tarm = []
tar_1 = 0.0
tar_2 = 0.0
tar_3 = 0.0
for tar in labels:
    if(tar == target_names[0]):
        tar_1 +=1
    if(tar == target_names[1]):
        tar_2 +=1
    if(tar == target_names[2]):
        tar_3 +=1
tarm.append(tar_1/len(labels))
tarm.append(tar_2/len(labels))
tarm.append(tar_3/len(labels))   
f = np.array(b)
def pr(feat,i,ff):
    count=0.0
    q = ff[:,i]
    for mm in q.flat:
        if(mm == feat):
            count = count + 1
    if(count!=0):
        return count/len(ff)
    else:
        return 1.0/(3+len(q))
index = 0
key=0
for row in features[s]:
    solution_p = 0
    solution_t = 0
    for i in range(0,3):
        p = 1
        is_type = (labels==target_names[i])
        ff = f[is_type]
        for j in range(0,4):
            p = p*pr(row[j],j,ff)
        p = p*tarm[i]
        if(p > solution_p):
            solution_p = p
            solution_t = i
        print "p(",i,"):",p
    print "真的值",target_names[target[s[index]]]
    print "预测值",target_names[solution_t]
    if(target[s[index]] == solution_t):
        
        key+=1
        index+=1       
print key
        
    


        
        

    
        
        
        
        


    
    
    
    
    
    
    
    






