# Gradient descent for Linear Regression 
# yhat = (weight)* input + bias
# MSE (loss function) = (y-yhat)**2 /N

# initialize parameters
import numpy as np 

x = np.random.randn(10,1)
y = 2*x + np.random.rand()

# paprameters
weight = 0.0
bias = 0.0
# hyperparameter
lr = 0.01

# create gradient descent funtion
def grad_descent(x,y,weight, bias, lr):
    dldw = 0.0
    dldb = 0.0
    N= x.shape[0]

    for xi, yi in zip(x,y):
        dldw += -2*xi*(yi - (weight*xi + bias))
        dldb += -2*(yi - (weight*xi + bias))
    # update parameters
    weight -= lr * (1/N) * dldw
    bias -= lr * (1/N) * dldb

    return weight, bias
# Iteratively make updates
for epoch in range(450):
    weight, bias = grad_descent(x,y,weight, bias, lr)
    yhat = weight * x + bias
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, parameter weight: {weight}, bias: {bias}')
