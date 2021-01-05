import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


lr=0.00001
n_iters=1000
weights=None
bias=None

def fit(X,y):
    n_samples,n_features=X.shape
    weights=np.zeros(n_features)
    list_err=[]
    list_iter=[]
    list_weights=np.zeros(3)
    bias=0
    for _ in range(n_iters):
        y_predicted=np.dot(X,weights)+bias
        mean_squared_error=(1/X.size)*np.sum(np.square(y_predicted-y))
        dw=(2/n_samples)*np.dot(X.T,(y_predicted-y))
        db=(2/n_samples)*np.sum(y_predicted-y)
        list_iter.append(_)
        list_err.append(mean_squared_error)
        weights-=lr*dw
        bias-=lr*db
        list_weights+=weights
        
        
    return bias,weights,list_iter,list_err,list_weights

df=pd.read_csv("Advertising.csv")
y=df.Sales
X=df[['TV', 'Radio', 'Newspaper']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=12)


b,w,list_iter,list_err,list_weights=fit(X_train,y_train)
def predict(X):
    predicted_y=np.dot(X,w)+b
    return predicted_y
predictions=predict(X_test)

print(list_weights)

plt.plot(list_iter,list_err)
plt.show()
