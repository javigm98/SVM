# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:54:33 2021

@author: javig
"""
from abc import abstractmethod
import numpy as np
from math import exp, tanh

# General class to implement kernel functions
class Kernel():
    def __init__(self):
        pass
    @abstractmethod
    def func(self, x, y):
        raise NotImplementedError
        
class Polynomial_kernel(Kernel):
    def __init__(self, c, d):
        super().__init__()
        self.c=c
        self.d=d
    def func(self, x, y):
        return (np.dot(x,y)+self.c)**self.d
    
class Gaussian_kernel(Kernel):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
    def func(self, x, y):
        return exp((-np.dot(x-y,x-y))/(2*self.sigma**2))

class Sigmoid_kernel(Kernel):
    def __init__(self, a,b):
        super().__init__()
        self.a=a
        self.b=b
    def func(self, x, y):
        return tanh(self.a*np.dot(x,y)+self.b)