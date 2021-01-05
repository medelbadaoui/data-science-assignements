
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np


#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
X=data[['Glucose']]
y=data['Outcome']

#Spliting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)

#Creating Model
h=LogisticRegression()
h.fit(X_train,y_train)


def predict(b,w,X):
    linear_model=np.dot(X,w)+b
    y_predicted=sigmoid(linear_model)
    y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
    return y_predicted_cls

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradientDescent(X,y):
    n_samples,n_features=X.shape
    learning_rate=0.001
    n_iterations=500
    bias=0
    W=np.zeros(n_features)
    for _ in range(n_iterations):
        linear_model=np.dot(X,W)+bias
        y_predicted=sigmoid(linear_model)
        d_W=(2/n_samples)*np.dot(X.T,(y_predicted-y)) 
        d_bias=(2/n_samples)*np.sum(y_predicted-y) 
        W-= learning_rate *d_W 
        bias-= learning_rate *d_bias 
    return bias,W

print(sigmoid(X_test))