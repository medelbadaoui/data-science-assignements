from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import preprocessing
import numpy as np
import seaborn as sns



#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
feature_cols = ['Glucose','BloodPressure','SkinThickness']
#feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
y=data[['Outcome']]



#plt.boxplot(X)

# for _ in range(0,3):
#     print(feature_cols[_])
#     print('Max : {}'.format(X[feature_cols[_]].max()))
#     print('Min : {}'.format(X[feature_cols[_]].min()))
#     print('Avg : {}'.format(X[feature_cols[_]].mean()))

#model from scratch  
def predict(b,w,X):
    linear_model=np.dot(X,w)+b
    y_predicted=sigmoid(linear_model)
    return np.array(y_predicted)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradientDescent(X,y):
    n_samples,n_features=X.shape
    learning_rate=0.0001        #0.0001         #0.01 #1e-8
    n_iterations=300      #3000      #20
    b=-1            #0           #2
    W=np.zeros((n_features,1),)
    loss_tab=[]
    for _ in range(n_iterations):
        y_predicted=predict(b,W,X)
        y_predicted=y_predicted.reshape(n_samples,1)
        y=np.array(y).reshape(n_samples,1)
        loss = (-1/n_samples )* np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
        loss_tab.append(loss)
        d_W=(2/n_samples)*np.dot(X.T,(y_predicted-y)) 
        d_bias=(2/n_samples)*np.sum(y_predicted-y) 
        W-= learning_rate *np.array(d_W)
        b-= learning_rate *d_bias 
    return b,W,loss_tab

#Spliting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)

b,w,ltab=gradientDescent(X_train,y_train)
y_predicted_cls=np.array([1 if i>0.5 else 0 for i in predict(b,w,X_test)])
print(y_predicted_cls)
#plt.plot(np.arange(1000),ltab,)

lg=LogisticRegression()
y_train=np.array(y_train)
lg.fit(X_train,y_train.ravel())


print(lg.predict(X_test))

y_test=np.array(y_test)

print(y_test.ravel())
print(metrics.accuracy_score(y_test,y_predicted_cls))
print(metrics.accuracy_score(y_test,lg.predict(X_test)))

#plt.plot(X_test,predict(b,w,X_test),'ro')
#plt.plot(X_test,y_test,'bo')
#plt.show()



