import numpy as np


y_real=np.array([0,0,0,0,1,1,1,1,1,1])
y_pred=np.array([0,0,0,1,1,0,1,1,0,1])


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

def precision(yr,yp):
    m=confmatrix(yr,yp)
    return (m.item(3)/(m.item(3)+m.item(1)))

def recall(yr,yp):
    m=confmatrix(yr,yp)
    return (m.item(3)/(m.item(3)+m.item(2)))

def f1score(yr,yp):
    return 2*((precision(yr,yp)*recall(yr,yp))/(precision(yr,yp)+recall(yr,yp)))
        
