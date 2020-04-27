# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:31:03 2019

@author: uchiha-ashish
"""

import pandas as pd
import numpy as np
import random
def load_data(path=None):
    f = np.loadtxt(path+'digitstrain.txt', delimiter = ',')
    data = pd.DataFrame(f)
    data = data.replace(0.000, int(0))        
    data = data.loc[:, (data != 0).any(axis=0)]
           
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return np.array(x), np.array(y)

X, Y = load_data('/home/tatras/Downloads/data/')
X = X.T
np.random.seed(99)


def sigmoid(z):
    t = 1/(1+np.exp(-z))
    return t

def softmax(z):
    e_x = np.exp(z)
    return e_x / e_x.sum(axis=0)

def softmax_backward(dA, cache):
    Z, Ww, Bb = cache
    s = np.exp(Z) / np.exp(Z).sum(axis=0)    
    dZ = dA*s*(1-s)
    return dZ  

def sigmoid_backward(dA, cache):
    Z = cache[0]   
    s = 1/(1+np.exp(-Z)) 
    dZ = dA*s   
    return dZ  

def initialize_parameters(layer_dim):
    parameters = {}
    L = len(layer_dim)    
    for l in range(1, L):    
        parameters["W"+str(l)] = np.random.randn(layer_dim[l], layer_dim[l-1])*0.1
        parameters["b"+str(l)] = np.random.randn(layer_dim[l], 1)*0.1
    return parameters

parameters = initialize_parameters(layer_dim = [659, 150, 10])

def linear_forward(inputs, W, b):
    cache1 = []    
    Z = np.dot(W, inputs) + b
    cache1 = [inputs, W, b]
    return Z, cache1

def activation_forward(A_prev, W, b, activation):  
    cache2 = []  
    activation_cache = []    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = [A, W, b]
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = softmax(Z)
        activation_cache = [A, W, b]
    cache2 = [linear_cache, activation_cache]
    return A, cache2

def forward_propagation(X, parameters):
    caches = []   
    L = len(parameters)//2
    A1 = X
    for l in range(L-1):
        A_prev = A1
        A1, cache3 = activation_forward(A_prev, parameters["W"+str(l+1)], parameters["b"+str(l+1)], activation = "sigmoid")
        caches.append(cache3)
    output, cache4 = activation_forward(A1, parameters["W"+str(L)], parameters["b"+str(L)], activation = "softmax")
    caches.append(cache4)    
    return output, caches

prediction, caches = forward_propagation(X, parameters)


def cost_function(output, value):
    cost = (value*np.log(output))
    cost = np.squeeze(-(1/m)*np.sum(cost))    
    return cost

m = Y.shape[0]
one_hot_target = np.eye(10)[(Y.astype(int))]
cost = cost_function(prediction.T, one_hot_target)


def linear_backward(dA, cache, dZ):
    A_prev, W, b = cache
    dW = np.dot(dZ, A_prev.T)
    db = dZ.sum(axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dA, linear_cache, dZ = dZZ)
    if activation == "softmax":        
        dZZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dA, linear_cache, dZ = dZZ)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    dAL = - (np.divide(one_hot_target.T, prediction) - np.divide(1 - one_hot_target.T, 1 - prediction))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax") 
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "sigmoid")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

grads = L_model_backward(prediction, one_hot_target, caches)

def update_parameters(parameters=None, grads=None, lr=None):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - lr * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - lr * grads["db" + str(l+1)]
    return parameters

parameters = update_parameters(parameters, grads, lr = 0.001)



for i in range(25):
    prediction, caches = forward_propagation(X, parameters)    
    print(cost_function(prediction.T, one_hot_target))
    grads = L_model_backward(prediction, one_hot_target, caches)
    parameters = update_parameters(parameters, grads, lr=0.0001)

    
    prediction_final, cah = forward_propagation(X, parameters)

    prediction_final = prediction_final.T
    pred_labels = np.zeros((3000, 1))
    for i in range(3000):
        pred_labels[i, 0] = np.argmax(prediction_final[i, :])

    Y = Y.reshape(3000, 1)
    counter = 0
    for i, j in zip(pred_labels, Y):
        if i == j:
            counter += 1
    accuracy = 100*counter/Y.shape[0]
    print(accuracy)
#79.16% accuracy at epoch 23
    





