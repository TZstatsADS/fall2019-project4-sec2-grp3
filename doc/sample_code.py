import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from math import exp

Rate = pd.read_csv("..\\data\\ml-latest-small\\ratings.csv")

list_user = Rate.userId.unique()
dict_user = {}
for i in range(len(list_user)):
    dict_user[list_user[i]] = i
    
list_movie = Rate.movieId.unique()    
dict_movie = {}
for i in range(len(list_movie)):
    dict_movie[list_movie[i]]=i
 
for i in range(len(Rate)):
    Rate["userId"][i] = dict_user[Rate["userId"][i]]
    Rate["movieId"][i] = dict_movie[Rate["movieId"][i]]

Train, Test = train_test_split(Rate, test_size = 0.2)

Train_user = list(Train["userId"])
Train_movie = list(Train["movieId"])
Train_rate = list(Train["rating"])
Test_user = list(Test["userId"])
Test_movie = list(Test["movieId"])
Test_rate = list(Test["rating"])

## A1

Factor = 30
Lambda = 0.1

P1 = np.random.normal(0, .1, (len(list_user),Factor))
Q1 = np.random.normal(0, .1, (len(list_movie),Factor))

def SGD(P, Q, epochs=300, lr=0.01):
    history = np.zeros((epochs,2))
    for _ in range(epochs):
        for i in range(len(Train)):
            user = Train_user[i]
            movie = Train_movie[i]
            error = Train_rate[i]-np.dot(Q[movie],P[user])
            temp = Q[movie]
            Q[movie] = Q[movie] + lr*(error*P[user]-Lambda*Q[movie])
            P[user] = P[user] + lr*(error*temp-Lambda*P[user])
        # calculate training and testing rmse after each epoch
        history[_,0] = sum([(Train_rate[i]-np.dot(Q[Train_movie[i]],P[Train_user[i]]))**2 for i in range(len(Train))])/len(Train)
        history[_,1] = sum([(Test_rate[i]-np.dot(Q[Test_movie[i]],P[Test_user[i]]))**2 for i in range(len(Test))])/len(Test)
        print(_,history[_])
    return history

result = SGD(P1,Q1)
##

## A2
## not converge sometimes

Factor = 20
Lambda_P = 0.2
Lambda_Q = 0.2

P2 = np.random.normal(0, .1, (len(list_user),Factor))
Q2 = np.random.normal(0, .1, (len(list_movie),Factor))

def GD(P, Q, epochs=300, lr=0.001):
    history = np.zeros((epochs,2))
    for _ in range(epochs):
        Gradient_P = np.zeros((len(list_user),Factor))
        Gradient_Q = np.zeros((len(list_movie),Factor))
        for i in range(len(Train)):
            user = Train_user[i]
            movie = Train_movie[i]
            error = Train_rate[i]-np.dot(Q[movie],P[user])
            Gradient_P[user] += error * Q[movie] 
            Gradient_Q[movie] += error * P[user]
        for i in range(len(list_user)):
            Gradient_P[i] -= Lambda_P * P[i]
        for i in range(len(list_movie)):
            Gradient_Q[i] -= Lambda_Q * Q[i]
        P = P + lr*Gradient_P
        Q = Q + lr*Gradient_Q
        # calculate training and testing rmse after each epoch
        history[_,0] = sum([(Train_rate[i]-np.dot(Q[Train_movie[i]],P[Train_user[i]]))**2 for i in range(len(Train))])/len(Train)
        history[_,1] = sum([(Test_rate[i]-np.dot(Q[Test_movie[i]],P[Test_user[i]]))**2 for i in range(len(Test))])/len(Test)
        print(_,history[_])
        # aotomatically change learning rate to avoid diverge
        if _>5 and history[_-1,0]-history[_,0]<0.1*lr:
            lr = lr/10
            print("learning rate change to ",lr)
    return history

result2 = GD(P2,Q2)
##

## Kernel Ridge Regression
#def my_kernel(X,Y):
#    return exp(2*(np.dot(X,Y.T)-1))

def KRR(Q,alpha=0.5,gamma=0.01):
    train_error = []
    test_error = []
    for i in Test.userId.unique():
        y = []
        X = []
        X_test = []
        y_test = []
        for j in range(len(Test)):
            if Test_user[j]==i:
                X_test.append(Q[Test_movie[j]]/np.linalg.norm(Q[Test_movie[j]]))
                y_test.append(Test_rate[j])
        for j in range(len(Train)):
            if Train_user[j]==i:
                X.append(Q[Train_movie[j]]/np.linalg.norm(Q[Train_movie[j]]))
                y.append(Train_rate[j])
        clf = KernelRidge(alpha = alpha, kernel = "rbf", gamma = gamma)
        clf.fit(X,y)
        y_estimate = clf.predict(X_test)
        y_train_estimate = clf.predict(X)
        train_error = train_error + [y[k] - y_train_estimate[k] for k in range(len(y))]
        test_error = test_error + [y_test[k] - y_estimate[k] for k in range(len(y_test))]
    print(sum([train_error[i]**2 for i in range(len(train_error))])/len(train_error))
    print(sum([test_error[i]**2 for i in range(len(test_error))])/len(test_error))
    return sum([test_error[i]**2 for i in range(len(test_error))])/len(test_error)

A1P3 = KRR(Q1,alpha=0.3)
A2P3 = KRR(Q2,alpha=0.3)
##