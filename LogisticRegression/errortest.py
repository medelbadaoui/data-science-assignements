import numpy as np
import pandas as pd


#Data collection
data = pd.read_csv(".\DataSets\diabetes.csv")

#Data initialization
feature_cols = ['Glucose','BloodPressure','SkinThickness']
#feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=data[feature_cols].to_numpy()
y=data[['Outcome']].to_numpy()

def sigmoid(input):    
    output = 1 / (1 + np.exp(-input))
    return output

def optimize(x, y,learning_rate,iterations,parameters): 
    size = x.shape[0]
    weight = parameters["weight"] 
    bias = parameters["bias"]
    for i in range(iterations): 
        sigma = sigmoid(np.dot(x, weight) + bias)
        loss = -1/size * np.sum(y * np.log(sigma) + (1 - y) * np.log(1-sigma))
        print(x.shape)
        dW = 2/size * np.sum(np.dot(x.T, (sigma - y)))
        print(dW)
        db = 2/size * np.sum(sigma - y)
        weight -= learning_rate * dW
        bias -= learning_rate * db 
    
    parameters["weight"] = weight
    parameters["bias"] = bias
    return parameters

init_parameters = {} 
init_parameters["weight"] = np.zeros(x.shape[1])
init_parameters["bias"] = 0

def train(x, y, learning_rate,iterations):
    parameters_out = optimize(x, y, learning_rate, iterations ,init_parameters)
    return parameters_out

parameters_out = train(x, y, learning_rate = 0.02, iterations = 10)