import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def mean(x):
    s=0
    for e in x:
        s+=e
    return s/len(x)
def sample_variance(x):
    m=mean(x)
    s=0
    for e in x:
         s+=math.pow(e-m,2)
    return s/(len(x)-1)
def pdf(x,sv,m):
    return 1/math.sqrt(2*math.pi*sv)*np.exp(-1/2*np.power(x-m,2)/sv)
#return (x-m)*(x-m)
def f(x):
    return x+1


#Data collection
data = pd.read_csv("NaiveBayes/miniiris.csv")
#Data initialization
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y=data['Species']


X_test=[4.9,3.0,4.9,2.1]

def probaclass(classe):
    return len(X[Y==classe])/len(X)

def psachantx(input,featureindex,classe):
    x1=np.array(X[feature_cols[featureindex]][Y==classe])
    sv=sample_variance(x1)
    moy=mean(x1)
    return pdf(input[featureindex],sv,moy)

def calculateproba(X_test):
    classes=['Iris-setosa','Iris-versicolor','Iris-virginica']
    pb=[]
    for i in classes :
        p=1
        for j in range(0,len(feature_cols)):
            p*=psachantx(X_test,j,i)
        pb.append((p*probaclass(i),i))
    return pb
        
probatable=calculateproba(X_test)        
y_predicted= max(probatable)   
print(probatable)
print(y_predicted)

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
gmodel=nb.fit(X,Y)
print(gmodel.predict([X_test]))





    
    
    