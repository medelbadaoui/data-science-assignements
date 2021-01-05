from math import  sqrt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from collections import Counter




#Data collection
data = pd.read_csv(".\DataSets\Iris.csv")
#Data initialization
feature_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
#feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y=data['Species']

from sklearn import preprocessing
#X = preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=12)

def normalization(X):
    Z=np.array(np.zeros(X.shape))
    min=X.min()
    max=X.max()
    Z=(X-min)/(max-min)
    return Z
def standarization(X):
    return (X-np.mean(X))/np.std(X)

#X_train = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])

#X_standarized=standarization(X)



def distances(X,input):  
    Sx=np.array(input)
    dist=[]
    for _ in range(0,len(X)):
        SX=X.iloc[_]   
        distance=sum((Sx-SX)**2)**0.5  
        dist.append(distance)
    return np.array(dist)


X_Input=[4.9,3.0,4.9,2.1]    

#print(distances(X,X_Input))

def predict(input,k):
    dist=distances(X,input)
    minvaluesindex=np.argpartition(dist, k)[:k]
    result=[]
    for _ in minvaluesindex:
        result.append(Y[_])
    return Counter(result).most_common(1)

import matplotlib.pyplot as plt

# plt.plot(X[y=='Iris-setosa'][feature_cols[2]],X[y=='Iris-setosa'][feature_cols[3]],'go',label='Iris-setosa')
# plt.plot(X[y=='Iris-versicolor'][feature_cols[2]],X[y=='Iris-versicolor'][feature_cols[3]],'ro',label='Iris-versicolor')
# plt.plot(X[y=='Iris-virginica'][feature_cols[2]],X[y=='Iris-virginica'][feature_cols[3]],'bo',label='Iris-virginica')

# plt.xlabel('Petal-Length')
# plt.ylabel('Petal-Width')
# plt.legend(loc="upper left")
# plt.show()

# plt.plot(X[y=='Iris-setosa'][feature_cols[0]],X[y=='Iris-setosa'][feature_cols[1]],'go',label='Iris-setosa')
# plt.plot(X[y=='Iris-versicolor'][feature_cols[0]],X[y=='Iris-versicolor'][feature_cols[1]],'ro',label='Iris-versicolor')
# plt.plot(X[y=='Iris-virginica'][feature_cols[0]],X[y=='Iris-virginica'][feature_cols[1]],'bo',label='Iris-virginica')

# plt.xlabel('Sepal-Length')
# plt.ylabel('Sepal-Width')

# plt.legend(loc="upper right")
# plt.show()

graf = plt.figure().gca(projection='3d')
def plotdata(x_label,y_label,z_label,classe,color,label):
    x = X[Y == classe ][x_label]
    y = X[Y == classe ][y_label]
    z = X[Y == classe ][z_label]
    graf.scatter(x,y,z,color=color, edgecolors='k',s=30, alpha=0.9, marker='o',label=label)
    graf.set_xlabel(x_label)
    graf.set_ylabel(y_label)
    graf.set_zlabel(z_label)
    graf.legend()

# plotdata(feature_cols[0],feature_cols[1],feature_cols[3],'Iris-virginica','g','Iris-virginica')
# plotdata(feature_cols[0],feature_cols[1],feature_cols[3],'Iris-versicolor','b','Iris-versicolor')
# plotdata(feature_cols[0],feature_cols[1],feature_cols[3],'Iris-setosa','r','Iris-setosa')
# plt.show()




#print('prediction =',predict(X_Input,7))





#print(X_Input)


#print(X_test)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
print(neigh.predict_proba([X_Input]))





# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
# def rateaccuracy():
#     neigh=None
#     accuracies=[]
#     for i in range(3,21):
#         neigh = KNeighborsClassifier(n_neighbors=i)
#         neigh.fit(X_train, y_train)
#         print('Accuracy k=',i,':',metrics.accuracy_score(y_test,neigh.predict(X_test)))
#         accuracies.append(metrics.accuracy_score(y_test,neigh.predict(X_test)))
#     plt.plot(range(3,21),accuracies,'ro')
#     plt.plot(range(3,21),accuracies)
#     plt.xlabel('k')
#     plt.ylabel('Accuracy')
# rateaccuracy()

# plt.show()



# print(class_freq)
# print(np.array(y_test).ravel())
