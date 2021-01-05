
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# X=np.array([1,2,3,4])
# y_real=np.array([1,2,3,4])

#Data collection
data = pd.read_csv("Advertising.csv")

#Data initialization
X=np.array(data['TV'])
y_real=np.array(data['Sales'])

#X=np.array([230.1,44.5,17.2,151.5,180.8,8.7,57.5,120.2,8.6,199.8])
#y_real=np.array([22.1,10.4,9.3,18.5,12.9,7.2,11.8,13.2,4.8,10.6])


def findW(X):
    n_iterations=50
    w0=0
    w1=0 #initialization
    learning_rate=0.000001
    list_w1=[]
    list_w0=[]
    list_mse=[]
    for i in range(n_iterations):
        y_predicted=w1*X+w0
        mean_squared_error=(1/X.size)*np.sum(np.square(y_predicted-y_real))
        print("iteration: {}, mse: {}".format(i,mean_squared_error))
        list_w1.append(w1)
        list_w0.append(w0)
        list_mse.append(mean_squared_error)
        dw1=-(2/X.size)*sum(X*(y_real-y_predicted))
        dw0=-(2/X.size)*sum(y_real-y_predicted)
        w1=w1-learning_rate*dw1
        w0=w0-learning_rate*dw0
    return list_w0,list_w1,list_mse
        
list_w0,list_w1,liste_mse=findW(X)
fig = plt.figure(facecolor='white')
fig.suptitle('Gradient Descent', fontsize=11)
fig.subplots_adjust(top=0.9,hspace=0.4,wspace=0.4)

fig.set_size_inches(10.5, 10.5, forward=True)
ax =fig.add_subplot(1, 2, 1)

plt.plot(list_w0,liste_mse)
plt.plot(list_w0,liste_mse,"ro")
plt.xlabel("W0")
plt.ylabel("MSE")
plt.title('MSe Decreasing in function of W0')

ax1=fig.add_subplot(1, 2, 2)
plt.plot(list_w1,liste_mse)
plt.plot(list_w1,liste_mse,"ro")
plt.xlabel("W1")
plt.ylabel("MSE")
plt.title('MSe Decreasing in function of W1')

plt.show()
