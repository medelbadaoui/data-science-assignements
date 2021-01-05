





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

#Data plotting
#plt.plot(X[y==1],y[y==1],'bo')
#plt.plot(X[y==0],y[y==0],'ro')

#Spliting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)

#Creating Model
h=LogisticRegression()
h.fit(X_train,y_train)

#Print weights and biais
print(h.coef_)
print(h.intercept_)

####
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

####

#Test model
y_predicted=h.predict(X_test)
#print(metrics.confusion_matrix(y_test,y_predicted))
print(confmatrix(y_test,y_predicted))
#print(metrics.accuracy_score(y_test,y_predicted))



print(y_test.size)
print(metrics.precision_recall_fscore_support(y_test,y_predicted))


#Plot sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
plt.plot(X_test,sigmoid(X_test),'rx')

def logit(x):
    return sigmoid(h.coef_*x+h.intercept_)
plt.plot(X_test,logit(X_test),'rx')


#ypr=logit([110,139,100,84,44])

#ypredicted=np.where(y_predicted > 0.5, 1, ypredicted)
#ypredicted=np.where(y_predicted < 0.5, 0, ypredicted)

#print(ypr)

#Plot Logit function
#plt.plot(X_test,y_predicted,'rx')


#plt.show()



