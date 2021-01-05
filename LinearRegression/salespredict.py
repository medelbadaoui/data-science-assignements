

from sklearn.linear_model import LinearRegression
from sklearn import metrics 
import matplotlib.pyplot as plt



#Data Inizialisation
X=[[230.1,37.8,69.2],[44.5,39.3,45.1],[17.2,45.9,69.3],
[151.5,41.3,58.5],[180.8,10.8,58.4],[8.7,48.9,75],[57.5,32.8,23.5],
[120.2,19.6,11.6],[8.6,2.1,1],[199.8,2.6,21.2]]
y=[22.1,10.4,9.3,18.5,12.9,7.2,11.8,13.2,4.8,10.6]

#Model creation
h=LinearRegression()

#Model Learning
h.fit(X,y)

#Printing Wi
print(h.intercept_)
print(h.coef_)

#Predicting Target for some Inputs
sales_predicted=h.predict([[66.1,5.8,24.2] ,[145,80,50]])

#Using only one variable from Data
X=[[i[0]] for i in X ]
h=LinearRegression()
h.fit(X,y)
sales_predicted=h.predict([[80],[220]])

#Plotting Inputs and real y
plt.plot(X,y,'ro')
#Plotting the Model
plt.plot([[0],[230.1]],h.predict([[0],[230.1]]))

plt.show()
#Mean Squared Error
y_predicted=h.predict(X)
y_real=y
metrics.mean_squared_error(y_predicted,y_real)