# -*- coding: cp936 -*-

import random

import math

train_data = []

test_data = []

def create_data( n , train_data):

    for num in range(1,n):
    
        x = random.uniform(0,1)

        x = round(x,3);
        
        y = math.sin(2*x*math.pi)+random.gauss(0,0.1)
        
        y = round(y,3)

        train_data.append((x,y))

def least_square_method(n , train_data , test_data):

    matrix = []

    x = []

    y = []

    for (a,b) in train_data:

        x.append(a)

        y.append(b)

    temp = 0

    for k in range(0,n+1):

        ratio = []

        sum = 0;

        number = temp;    

        label = 0;
        
        for i in range(0,n+1):
           
            for j in range(0,len(train_data)):

                sum = sum + math.pow(x[j],number)

                if i == 0 :

                    label = label + y[j] * math.pow(x[j],number)

            label = round(label,3)

            sum = round(sum,3)
            
            ratio.append(sum)

            sum = 0

            number = number + 1

        ratio.append((-1) * label)

        matrix.append(ratio)

        temp = temp + 1

        label = 0

    for k in range(0,len(matrix)-1):
                   

        for i in range(k+1 , len(matrix)):

            temp = matrix[k][k] / (matrix[i][k])

            matrix[i][k] = 0 

            for j in range(k+1,len(matrix[i])):

                 matrix[i][j] = matrix[k][j] - temp*matrix[i][j]

    solution = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0 , len(matrix)):

        j = len(matrix) - i - 1

        temp = (matrix[j][len(matrix)])*(-1)

        for k in range(j+1 , len(matrix)):

            temp =  temp - matrix[j][k]*solution[k]

        solution[j] = temp/matrix[j][j]

    E = 0;

    for i in range(0,len(x)):

        sum = 0

        for j in range(0,len(matrix)):

            sum = sum + solution[j]* math.pow(x[i],j)

        E = E + math.pow(y[i] - sum , 2)

    print "leaing data误差:",(1.0/2)*E

    E = 0

    sum = 0

    for (a,b) in test_data:

        for i in range(0,len(matrix)):

            sum = sum + math.pow(a,i)*solution[i]

        E = E + (b-sum)*(b-sum)

        sum = 0

    print "test data error%:",(1.0/2)*E

def gradient_decent(n , train_data , test_data , stemp):

    solution = []

    x = []

    y = []

    for (a,b) in train_data:

        x.append(a)

        y.append(b)
        
    for i in range(0,n+1):

        solution.append(0.5)

    E = 0

    for i in range(0,len(x)):

            sum = 0

            for j in range(0,len(solution)):

                sum = sum + math.pow(x[i],j)*solution[j]

            E = E + (y[i] - sum) * (y[i] - sum)

    sum = 0

    E = E + sum*(1.0/2)

    temp_1 = 0

    for i in range(0,n+1):

        temp_1 = temp_1 + math.pow(solution[i],2)

    E = E + temp_1*(1/2)*math.exp(-18)

    print E

    while 1==1:

        solution_new = []

        for i in range(0,n+1):

            sum = 0

            temp = 0

            for k in range(0,len(train_data)):

                temp = 0
                
                for j in range(0,n+1):

                    temp =  temp + math.pow(x[k] , j) * solution[j]

                sum = sum + (temp - y[k]) * math.pow(x[k],i)

            solution_new.append(solution[i] - stemp*sum - math.exp(-18)*solution[i])

        E_new = 0

        for i in range(0,len(x)):

            sum = 0

            for j in range(0,len(solution)):

                sum = sum + math.pow(x[i],j)*solution_new[j]

            E_new = E_new + (y[i] - sum) * (y[i] - sum)

        temp_2 = 0

        for i in range(0,n+1):

            temp_2 = temp_2 + math.pow(solution_new[i],2)

        E_new = E_new + temp_2*(1/2)*math.exp(-18)

        print E_new

        if E - E_new < 0.0000001 :

            solution = solution_new

            E = E_new

            break

        else:

            E = E_new

            for i in range(0,len(solution)):

                solution[i] = solution_new[i]

            solution_new = []

    for i in range(0,len(x)):

        print x[i],

        sum = 0

        for j in range(0,len(solution)):

            sum = sum + math.pow(x[i],j) * solution[j]

        print sum

    E_test = 0

    for (a,b) in test_data:

        sum = 0

        for i in range(0,len(solution)):

            sum = sum + solution[i]*math.pow(a,i)

        E_test = E_test + (1.0/2)*math.pow((sum-b),2)

    for i in range(0,len(solution)):

        E_test = E_test + (1.0/2)*math.exp(-18)*solution[i]*solution[i]

    print "多项式次数:",n

    print "leaning data error:",E

    print "testing data error:",E_test

def first_order(n,matrix,ratio,solution):

    a = []

    for i in range(0,len(matrix)):

        sum = 0

        for j in range(0,len(matrix[i])):

            sum = sum + matrix[i][j]*solution[j]

        a.append(sum)

    #for i in range(0,len(ratio)):

     #   a[i] = a[i] - ratio[i]

    return a
            
def mod_height(first_ord):

    sum = 0

    for i in range(0,len(first_ord)):

        sum = sum + math.pow(first_ord[i],2)

    return math.sqrt(sum)

def calculate_p(n,p_old,first_ord,beta):

    temp = []

    for i in range(0,len(first_ord)):

        temp.append((-1)*first_ord[i])

    for i in range(0,len(first_ord)):

        temp[i] = temp[i] + beta*p_old[i]

    return temp

def step(n,first_ord,p_new,matrix):

    fenzi = 0.0

    for i in range(0,n):

        fenzi = fenzi + first_ord[i]*p_new[i]

    temp = []

    for i in range(0,n):

        sum = 0

        for j in range(0,n):

            sum = sum + matrix[j][i]*p_new[j]

        temp.append(sum)

    fenmu = 0.0

    for i in range(0,n):

        fenmu = fenmu + temp[i]*p_new[i]

    return (-1)*fenzi/fenmu

def calculate_beta(n,first_ord_temp,first_ord):

    fenzi = 0

    for i in range(0,n):

        fenzi = fenzi + first_ord_temp[i]*first_ord_temp[i]

    fenmu = 0

    for i in range(0,n):

        fenmu = fenmu + first_ord[i]*first_ord[i]

    return fenzi/fenmu
        
def conjugate_gradient(n,train_data,test_data):

    solution = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

    x = []

    y = []

    for (a,b) in train_data:

        x.append(a)

        y.append(b)

    matrix = []

    temp = []

    for i in range(0,n):

        for j in range(0,n):

            sum = 0

            for k in range(0,len(train_data)):

                sum = sum + math.pow(x[k],i+j)

            temp.append(sum)

        matrix.append(temp)

        temp = []

    for i in range(0,len(matrix)):

        for j in range(0,len(matrix[i])):

             print matrix[i][j],

    for i in range(0,n):

        matrix[i][i] = matrix[i][i] + math.exp(-18)

    ratio = []

    for i in range(0,n):

        sum = 0

        for j in range(0,len(train_data)):

            sum = sum + y[j]*math.pow(x[j],i)

        ratio.append(sum)

    for i in range(0,n):

        print ratio[i],

    first_ord = []

    #solution1 = [-2,4]

    #matrix1 = [[3,-1],[-1,1]]

    #print matrix1[0][0]

    #ratio1 = [-2,0]

    first_ord = first_order(n,matrix,ratio,solution)

    first_ord_temp = []

    p_old = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    p_new = []

    beta = 0
    
    while mod_height(first_ord) > 0.0001:

        p_new = calculate_p(n,p_old,first_ord,beta)

        stepp = step(n,first_ord,p_new,matrix)

        E = 0

        #for (a,b) in train_data:

         #   sum = 0

          #  for i in range(0,n):

           #     sum = sum + math.pow(a,i)*solution[i]

            #E = E + (b-sum)*(b-sum)

        # print E
        
        print "-----"

        print stepp

        for i in range(0,len(p_new)):

            print p_new[i],

        for i in range(0,n):

            solution[i] = solution[i] + stepp*p_new[i]

        print "%%%%"

        for i in range(0,n):

            print solution[i],

        print "****"

        first_ord_temp = first_order(n,matrix,ratio,solution)

        beta = calculate_beta(n,first_ord_temp,first_ord)

        first_ord = first_ord_temp

        p_old = p_new
    
            
create_data(100,train_data)

for (a,b) in train_data:

    print a,b

create_data(10,test_data)

#gradient_decent(5,train_data,test_data,math.pow(0.1,2))

#gradient_decent(6,train_data,test_data,math.pow(0.1,1))

#gradient_decent(4,train_data,test_data,math.pow(0.1,2))

#gradient_decent(3,train_data,test_data,math.pow(0.1,1))

#gradient_decent(1,train_data,test_data,math.pow(0.1,2))

#gradient_decent(2,train_data,test_data,math.pow(0.1,2))

#gradient_decent(7,train_data,test_data,math.pow(0.1,2))

#gradient_decent(10,train_data,test_data,math.pow(0.1,2))

#gradient_decent(9,train_data,test_data,math.pow(0.1,2))

#gradient_decent(8,train_data,test_data,math.pow(0.1,2))

#conjugate_gradient(2,train_data,test_data)

for i in range(0,9):

    print "多项式次数",i
    
    least_square_method(i,train_data,test_data)
