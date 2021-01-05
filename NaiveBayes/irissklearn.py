
from math import  sqrt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB





#Data collection
data = pd.read_csv(".\DataSets\Iris.csv")
#Data initialization
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y=data['Species']

from sklearn import preprocessing

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=12)


nb=BernoulliNB()
bmodel=nb.fit(X_train,y_train)

nb=GaussianNB()
gmodel=nb.fit(X_train,y_train)

nb=MultinomialNB()
mmodel=nb.fit(X_train,y_train)

knn=KNeighborsClassifier(n_neighbors=12)
kmodel=knn.fit(X_train,y_train)

print('KNN :',metrics.accuracy_score(y_test,kmodel.predict(X_test)))
print('GAUSSIAN NAIVE BAYES : ',metrics.accuracy_score(y_test,gmodel.predict(X_test)))
print('BERNOULLI NAIVE BAYES : ',metrics.accuracy_score(y_test,bmodel.predict(X_test)))
print('MULTINOMIAL NAIVE BAYES : ',metrics.accuracy_score(y_test,mmodel.predict(X_test)))
