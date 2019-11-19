import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

Rate = pd.read_csv("..\\data\\ml-latest-small\\ratings.csv")

## relabel userId and movieId
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

## Train, Test split
Train, Test = train_test_split(Rate, test_size = 0.1)

Train_user = list(Train["userId"])
Train_movie = list(Train["movieId"])
Train_rate = list(Train["rating"])
Test_user = list(Test["userId"])
Test_movie = list(Test["movieId"])
Test_rate = list(Test["rating"])

## A1
def A1(Factor=15, Lambda=0.1, epochs=300, lr=0.01):
    P = np.random.normal(0, .1, (len(list_user),Factor))
    Q = np.random.normal(0, .1, (len(list_movie),Factor))
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
        history[_,0] = (sum([(Train_rate[i]-np.dot(Q[Train_movie[i]],P[Train_user[i]]))**2 for i in range(len(Train))])/len(Train))**.5
        history[_,1] = (sum([(Test_rate[i]-np.dot(Q[Test_movie[i]],P[Test_user[i]]))**2 for i in range(len(Test))])/len(Test))**.5
        print(_,history[_])
        if _>50 and history[_-1,0]-history[_,0]<0.1*lr and lr>0.0001:
            lr = lr/10
            print("learning rate change to ",lr)
    return P,Q,history

result1 = A1()

## A2
def A2(Factor=3, Lambda_P=0.2, Lambda_Q=0.1, epochs=500, lr=0.001):
    P = np.random.normal(0, .1, (len(list_user),Factor))
    Q = np.random.normal(0, .1, (len(list_movie),Factor))
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
        history[_,0] = (sum([(Train_rate[i]-np.dot(Q[Train_movie[i]],P[Train_user[i]]))**2 for i in range(len(Train))])/len(Train))**.5
        history[_,1] = (sum([(Test_rate[i]-np.dot(Q[Test_movie[i]],P[Test_user[i]]))**2 for i in range(len(Test))])/len(Test))**.5
        print(_,history[_])
        # aotomatically change learning rate to avoid diverge
        if _>50 and history[_-1,0]-history[_,0]<0.1*lr and lr>0.00001:
            lr = lr/10
            print("learning rate change to ",lr)
    return P,Q,history

result2 = A2()

## Kernel Ridge Regression
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
    print((sum([train_error[i]**2 for i in range(len(train_error))])/len(train_error))**.5)
    print((sum([test_error[i]**2 for i in range(len(test_error))])/len(test_error))**.5)
    return (sum([test_error[i]**2 for i in range(len(test_error))])/len(test_error))**.5

A1P3 = KRR(result1[1],alpha=0.2,gamma=0.004)
A2P3 = KRR(result2[1],alpha=0.2,gamma=0.002)