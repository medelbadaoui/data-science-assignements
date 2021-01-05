import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB



# dataX =[[0,1,1,0,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],[1,0,0,1,0],[0,1,0,1,0],[1,1,1,1,1],[1,1,0,1,0],[1,1,1,0,1],[1,0,1,0,1]]
# dataY=[['c1'],['c1'],['c1'],['c1'],['c1'],['c2'],['c2'],['c2'],['c2'],['c2']]
# dfx = pd.DataFrame(dataX,columns=['x1','x2','x3','x4','x5'])
# dfy= pd.DataFrame(dataY,columns=['label'])
# data=dfx.join(dfy)
# features=['x1','x2','x3','x4','x5']
# X=data[features]
# y=data['label']









#print(len(X[y=='c1']))



#X_test=[[0,0,1,1,1]]
X_test=[[0,0,1,1,1]]



def calculateproba(x):
    Xc1=X[y=='c1']
    Xc2=X[y=='c2']
    pxc1=1
    pxc2=1
    tmp=1
    lp=[]
    for _ in range(0,len(features)):
        #p=len(X[X['x'+str(tmp)]==x[0][_]])
        pxc1*=len(Xc1[Xc1['x'+str(tmp)]==x[0][_]])/len(Xc1) * len(Xc1)/len(X) 
        pxc2*=len(Xc2[Xc2['x'+str(tmp)]==x[0][_]])/len(Xc2) * len(Xc2)/len(X) 
        tmp+=1
    lp=[[pxc1,'c1'],[pxc2,'c2']]
    return lp
    
    


y_predicted=calculateproba(X_test)
print(y_predicted)

#print(data['x1'])
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

gnb = BernoulliNB()
y_pred = gnb.fit(X, y).predict(X_test)
print(y_pred)
