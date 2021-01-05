
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn import preprocessing

#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
feature_cols = ['Age','Glucose']
#feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
y=data[['Outcome']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)


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
def sigmoid(x):
    return 1/(1+np.exp(-x))
def slope(w,b,X):
  return (-b/w[1])-(w[0]/w[1])*X
def predict(bias,W,X):
    linear_model=np.dot(X,W)+bias
    y_predicted=sigmoid(linear_model)
    y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
    return np.array(y_predicted),np.array(y_predicted_cls)
def fit(X,y):
    n_samples,n_features=X.shape
    learning_rate=0.001        #0.0001         #0.01 #1e-8
    n_iterations=10000     #3000      #20
    b=0          #0           #2
    W=np.zeros((n_features,1),)
    for _ in range(n_iterations):
        y_predicted,y_predicted_cls=predict(b,W,X)
        y_predicted=y_predicted.reshape(n_samples,1)
        y_predicted_cls=y_predicted_cls.reshape(n_samples,1)
        
        y=np.array(y).reshape(n_samples,1)
        
        loss = (-1/n_samples )* np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
        
        d_W=(1/n_samples)*np.dot(X.T,(y_predicted_cls-y)) 
        d_bias=(1/n_samples)*np.sum(y_predicted_cls-y) 
        
        W-= learning_rate *np.array(d_W)
        b-= learning_rate *d_bias
        acc=metrics.accuracy_score(y,y_predicted_cls)
        if(_%1000==0):
            print(W,b,loss,acc) 
             #Data plotting
            plt.plot(X[y==1][feature_cols[0]],X[y==1][feature_cols[1]],'bo')
            plt.plot(X[y==0][feature_cols[0]],X[y==0][feature_cols[1]],'gD')
            
            min=X[feature_cols[0]].min()
            max=X[feature_cols[0]].max()
            print(min,max)
            slope1 = slope(W,b,min)
            slope2 = slope(W,b,max)
            plt.plot([min,slope1], [max,slope2],'r')
            for input, prediction, label in zip(X, y_predicted_cls, y):
                if prediction is not label:
                   plt.plot(input[0],input[1], 'rx')
                    
                    
            plt.show()
    return b,W

b,w=fit(X_train,y_train)

