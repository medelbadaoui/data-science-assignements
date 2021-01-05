import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("Advertising.csv")


X=[data[['TV']],data[['Radio']],data[['Newspaper']]]
y_real=data[['Sales']]

def predict(X,learninrate):
    b=np.zeros(3)
    W=np.zeros(3)
    print(W)
    n=len(X)
    y_predicted=b+np.dot(X,W)
    dw=(2/n)*np.dot(X.T,(y_predicted-y_real))
    W=W-learninrate*dw
    print(W)
predict(X,0.000001)     
