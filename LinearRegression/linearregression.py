from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics 
import numpy as np


#Data collection
data = pd.read_csv("Advertising.csv")

#Data initialization
X=data[['TV']]
y=data[['Sales']]

#Creating Model
h=LinearRegression()
h.fit(X,y)

#Plotting Inputs and real y
plt.plot(X,y,'ro')

#Plotting the Model
min = X.min().values[0] 
max = X.max().values[0] 
plt.plot([[min],[max]],h.predict([[min],[max]]))

#Mean Squared Error & Mean Absolute Error
y_predicted=h.predict(X)
y_real=y
mean_squared_error=metrics.mean_squared_error(y_predicted,y_real)
mean_absolute_error=metrics.mean_absolute_error(y_predicted,y_real)

#MSE & MAE from Scratch
calculatedmse=(1/X.size)*np.sum(np.square(y_predicted-y_real))
calculatedmae=np.sum(np.abs(y_predicted-y_real))/X.size

#Accuracy
accuracy=h.score(X,y)

#Printing MSE , MAE and accuracy 
print("mse: {} calculated mse: {}".format(mean_squared_error,np.float(calculatedmse)))
print("mae: {} calculated mae: {}".format(mean_absolute_error,np.float(calculatedmae)))
print("accuracy : {:.2f}% ".format(accuracy*100))

#showing the plot
plt.show()


