import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np



#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
#feature_cols = ['Glucose','BloodPressure','SkinThickness']
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X=data[feature_cols]
#X=preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
Y=data['Outcome']

#X = preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=12)

from sklearn.tree import DecisionTreeClassifier
#create the Decision Tree Classifier model
model = DecisionTreeClassifier()
# Train Decision Tree Classifer
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))
