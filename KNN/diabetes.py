
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import preprocessing
import numpy as np



#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
#feature_cols = ['Glucose','BloodPressure','SkinThickness']
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y=data['Outcome']

from sklearn import preprocessing
#X = preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=12)




def plot2Ddata(x_label,y_label,classe,color,label):
    x = X[Y == classe ][x_label]
    y = X[Y == classe ][y_label]
    plt.plot(x,y,color+'o',label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
plot2Ddata(feature_cols[2],feature_cols[3],0,'g','not sick')
plot2Ddata(feature_cols[2],feature_cols[3],1,'b','sick')   

graf = plt.figure().gca(projection='3d')
def plot3Ddata(x_label,y_label,z_label,classe,color,label):
    x = X[Y == classe ][x_label]
    y = X[Y == classe ][y_label]
    z = X[Y == classe ][z_label]
    graf.scatter(x,y,z,color=color, edgecolors='k',s=30, alpha=0.9, marker='o',label=label)
    graf.set_xlabel(x_label)
    graf.set_ylabel(y_label)
    graf.set_zlabel(z_label)
    graf.legend()

 
# plot3Ddata(feature_cols[0],feature_cols[1],feature_cols[3],0,'g','not sick')
# plot3Ddata(feature_cols[0],feature_cols[1],feature_cols[3],1,'b','sick')


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
def rateaccuracy():
    neigh=None
    accuracies=[]
    for i in range(1,31):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(X_train, y_train)
        print('Accuracy k=',i,':',metrics.accuracy_score(y_test,neigh.predict(X_test)))
        accuracies.append(metrics.accuracy_score(y_test,neigh.predict(X_test)))
    plt.plot(range(1,31),accuracies,'ro')
    plt.plot(range(1,31),accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
rateaccuracy()

plt.show()



