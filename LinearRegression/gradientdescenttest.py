import numpy as np
import matplotlib.pyplot as plt


X=np.array([1,2,3,4])
y_real=np.array([1,2,3,4])
def findW(X):
    n_iterations=21
    w0=0 
    w1=0 #initialization
    list_w=[]
    list_mse=[]
    for i in range(n_iterations):
        y_predicted=w1*X+w0
        mean_squared_error=(1/X.size)*np.sum(np.square(y_predicted-y_real))
        print("iteration: {}, y_predicted:{}, mse: {}".format(i,y_predicted,mean_squared_error))
        list_w.append(w1)
        list_mse.append(mean_squared_error)
        w1=w1+0.1
    return list_w,list_mse
        

w1_list,mse_list=findW(X)

plt.plot(w1_list,mse_list)
plt.plot(w1_list,mse_list,"ro")

plt.show()
