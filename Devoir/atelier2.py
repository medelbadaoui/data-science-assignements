from random import seed
from random import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 


def sigmoid(x):
    return 1/(1+np.exp(-x))
def confmatrix(yr,yp):
    tp,tn,fp,fn=0,0,0,0
    for (i,j) in zip(yr, yp):
        if(i==j==0):
            tn+=1
        if(i==j==1):
            tp+=1
        if(i<j):
            fp+=1
        if(i>j):
            fn+=1
    return np.matrix([[tn,fp],[fn,tp]])
def accuracy(yr,yp):
    m=confmatrix(yr,yp)
    return (m.item(0)+m.item(3))/m.sum()



def generateData():
  seed(1)
  min=0
  max=5
  dataset=[]
  for _ in range(100):
    valueX1 = random()
    scaledvalueX1 = min + (valueX1 * (max - min))
    valueX2 = random()
    scaledvalueX2 = min + (valueX2 * (max - min))
    x1=round(scaledvalueX1,1)
    x2=round(scaledvalueX2,1)
    y0=-x1+5
    cl=0
    if(x2>y0):
      cl=1
    dataset.append([x1,x2,cl])
  return np.array(dataset)


data=generateData()
X_train,X_test,y_train,y_test=train_test_split(data[:,0:2],data[:,2],test_size=0.2, random_state=12)


def plotdata():
  plt.plot(data[:,0][data[:,2]==1],data[:,1][data[:,2]==1], 'bo')
  plt.plot(data[:,0][data[:,2]==0],data[:,1][data[:,2]==0], 'gD')
  

def slope(w,b,X):
  return (-b/w[1])-(w[0]/w[1])*X

def predict(bias,W,X):
  linear_model = np.dot(X,W) + bias
  y_predicted=sigmoid(linear_model)
  y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
  return y_predicted_cls

lr=0.001
n_iters=100000
def fit(X,y):
    bias=0
    W=np.array([0.,0.])
    n_samples,n_features=X.shape
    for i in range(n_iters):
        linear_model=np.dot(X,W)+bias
        y_predicted=sigmoid(linear_model)
        dW=(1/n_samples)*np.dot(X.T,(y_predicted-y))
        dbias=(1/n_samples)*np.sum(y_predicted-y)
        loss = (-1/n_samples)* np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
        #dbias=1/n_samples*np.sum(y_predicted*(1-y_predicted))
        #dW=1/n_samples*np.sum(np.dot(y_predicted*(1-y_predicted),X))
        W-=lr*dW #ligne12
        bias-=lr*dbias #ligne13
        y_predicted_cls=np.array([1 if i>0.5 else 0 for i in y_predicted])
        acc=accuracy(y_predicted_cls,y)
        if(i%10000==0):
          print(i,W,bias,loss,acc)
          min=X.min()
          max=X.max()
          slope1 = slope(W,bias,min)
          slope2 = slope(W,bias,max)
          plt.plot([min,slope1], [max,slope2])
          plotdata()
          predictions = predict(bias,W,X)
          for input, prediction, label in zip(X, predictions, y):
            if prediction != label:
              plt.plot(input[0],input[1], 'rx')
          plt.show()
    return bias,W
  
b,W=fit(X_train,y_train)



