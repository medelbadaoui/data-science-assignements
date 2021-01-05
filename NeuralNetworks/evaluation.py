#Exercice 1
# 1- Gradient Descent est un algorithme utilisé pour trouver les meilleurs
#Wi (weights) et le biais (w0) en minimisant cost function (MSE,MAE ...)
#
#Exercice 2
#1-Binary cross Entropy
#2- on fait le choix de cost function ça depends sur le problems qu'on a par exemple s'il s'agit de regression on peut
#utiliser MSE ou MAE , pour la classification on utilise  Cross entropy.
#3-
#-Mean Squared Error (2 au lieu de 1 )
#-



import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 

# #Data collection
# data = pd.read_csv(".\DataSets\diabetes.csv")

# #Data initialization
# feature_cols = ['Glucose','BloodPressure','SkinThickness']
# #feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
# X=data[feature_cols]
# #X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
# y=data[['Outcome']]

# #Spliting data
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)



#exercice 3 - 1
from random import seed
from random import random
def f(x):
    return -x+5
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
    y0=f(x1)
    cl=0
    if(x2>y0):
      cl=1
    dataset.append([x1,x2,cl])

  return dataset

data=np.array(generateData())
print(data)

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



#plt.plot([0,g(bias,w1,w2,0)],[0,g(bias,w1,w2,5)])

X_train,X_test,y_train,y_test=train_test_split(data[:,0:2],data[:,2],test_size=0.2, random_state=12)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def plotData(dataset):
    c0=dataset[:,2]==0
    c1=dataset[:,2]==1
    XD1=dataset[:,0][c0]
    YD1=dataset[:,1][c0]
    XD2=dataset[:,0][c1]
    YD2=dataset[:,1][c1]
    plt.plot(XD1,YD1, 'gD')
    plt.plot(XD2,YD2, 'bo')
    plt.show()
    

lr=0.001
n_iters=100000

def predict(bias,W,X):
    linear_model=np.dot(X,W)+bias
    y_predicted=sigmoid(linear_model)
    y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
    return y_predicted_cls
def fit(X,y):
    bias=0
    W=np.array([0.,0.])
    n_samples,n_features=X.shape
    for i in range(n_iters):
        linear_model=np.dot(X,W)+bias
        y_predicted=sigmoid(linear_model)
        
        dW=(1/n_samples)*np.dot(X.T,(y_predicted-y))
        dbias=(1/n_samples)*np.sum(y_predicted-y)
        W-=lr*dW #ligne12
        bias-=lr*dbias #ligne13
        
        loss = (-1/n_samples)* np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
        y_predicted_cls=np.array([1 if i>0.5 else 0 for i in y_predicted])
        acc=accuracy(y,y_predicted_cls)
        if(i%10000==0):
            print(i,W,bias,loss,acc)
            w1=W[0]
            w2=W[1]
            slope = (-bias/w2)-(w1/w2)*X_test
            plt.plot(X_test, slope,'r')
            plotData(data)
    return bias,W


b,w=fit(X_train,y_train)






    
    
   




#Data plotting
#plt.plot(X[y==1],y[y==1],'bo')
#plt.plot(X[y==0],y[y==0],'ro')