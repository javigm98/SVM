# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:53:34 2021

@author: javig
"""

from abc import abstractmethod
import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import SVC

# General SVM predictor class. Particular implementations inherite from this class
class SVM_predictor():
    # Defaul constructor
    def __init__(self):
        self.b=0
        self.w = None
    @abstractmethod
    def train(self, train_data, trainlabels, C):
        raise NotImplementedError
        
    # Returns sign(w*data+b) as the label for data
    def predict(self, data):
        return np.sign(np.matmul(self.w.T, data.T)+self.b)
    
    # For each point in data return its prediction and concatenate all of them
    def predict_multiple(self, data):
        predictions= np.array([])
        for x in data:
            predictions = np.append(predictions, self.predict(x))
        return np.array([predictions]).T
    
    # Returns a vector with value of w*x+b for each x in xy
    def decision_function(self, xy):
        P=np.array([])
        for i in range(xy.shape[0]):
           P=np.append(P,np.dot(xy[i], self.w)+self.b) 
        return P

# SVM algorithm solving the primal optimization problem
class SVM_predictor_primal(SVM_predictor):
    def __init__(self):
        super().__init__()

    def train(self,traindata, trainlabels, C):
        m = np.shape(traindata)[0]
        n = np.shape(traindata)[1]

        P=matrix(np.block([[np.eye(n), np.zeros((n,m+1))],[np.zeros((m+1,n)), np.zeros((m+1,m+1))]]))
        q = matrix(np.vstack([np.zeros((n,1)),0,np.ones((m,1))*C]))

        aux = np.eye(m)
        for i in range(0,m):
            aux[i][i]=trainlabels[i]

        G=matrix(np.vstack([np.hstack([np.matmul(aux,traindata)*-1, trainlabels*-1, np.eye(m)*-1]),np.hstack([np.zeros((m,n)),np.zeros((m,1)), -np.eye(m)])]))
        h=matrix(np.vstack([-np.ones((m,1)),np.zeros((m,1))]))
        
        solvers.options['show_progress'] = False
        solution=solvers.qp(P,q,G,h)

        self.w=np.array(solution['x'][0:n])
        self.b=solution['x'][n]

# SVM algorithm solving the dual optimization problem
class SVM_predictor_dual(SVM_predictor):
    def __init__(self):
        super().__init__()
    
    def train(self, traindata, trainlabels, C):
        
        m = np.shape(traindata)[0]
        n = np.shape(traindata)[1]

        P=matrix(np.multiply(np.matmul(trainlabels, trainlabels.T), np.matmul(traindata, traindata.T)))
        P=matrix(P, P.size, 'd')
        q=matrix(-np.ones((m,1)))
        A=matrix(trainlabels.T)
        A=matrix(A, A.size, 'd')
        b=matrix(0.0)
        G=matrix(np.vstack([-np.eye(m), np.eye(m)]))
        h=matrix((np.hstack([np.zeros(m), np.ones(m)*C])).T)

        solvers.options['show_progress'] = False

        solution=solvers.qp(P,q,G,h,A,b)

        u=np.array(solution['x'])

        sv_ind= (u>1e-4).flatten()
        sv_ind_leqC=(u<C).flatten()
        sv_ind_mh=np.logical_and(sv_ind, sv_ind_leqC)

        self.w = ((trainlabels * u).T @ traindata).reshape(-1,1)
        self.b = np.mean(trainlabels[sv_ind_mh] - np.dot(traindata[sv_ind_mh], self.w))

# SVM algorithm using sklearn implementation (sklearn.svm.SVC)
class SVM_predictor_sklearn(SVM_predictor):
    def __init__(self):
        super().__init__()
    
    # We choose kernel='linear' so that no kernel function is used
    def train(self, traindata, trainlabels, C):
        self.svm_clf=SVC(kernel='linear', C=C)
        trainlabels = np.ravel(trainlabels)
        # Inestead of saving the values of w and b, we store the whole classifier
        # in order to use other functionalities later
        self.svm_clf.fit(traindata, trainlabels)
    
    # We use the SVC class predict function
    def predict_multiple(self, data):
        return np.array([self.svm_clf.predict(data)]).T
    
    # We use the SVC class decision function
    def decision_function(self, xy):
        return self.svm_clf.decision_function(xy)

# SVM algoritm using kernel functions to compute scalar products in the dual problem
class SVM_predictor_kernel(SVM_predictor):
    # We need to provide a kernel instace to the constructor
    def __init__(self, kernel):
        super().__init__()
        self.kernel=kernel
    
    def train(self, traindata, trainlabels, C):
        m = np.shape(traindata)[0]
        n = np.shape(traindata)[1]

        self.m = m
        self.n = n
        self.traindata = traindata
        self.trainlabels= trainlabels

        K=np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                K[i,j]=self.kernel.func(traindata[i], traindata[j])

        P=matrix(np.multiply(np.matmul(trainlabels, trainlabels.T), K))
        P=matrix(P, P.size, 'd')
        q=matrix(-np.ones((m,1)))
        A=matrix(trainlabels.T)
        A=matrix(A, A.size, 'd')
        b=matrix(0.0)
        G=matrix(np.vstack([-np.eye(m), np.eye(m)]))
        h=matrix((np.hstack([np.zeros(m), np.ones(m)*C])).T)

        solvers.options['show_progress'] = False
        solution=solvers.qp(P,q,G,h,A,b)

        u=np.array(solution['x'])
        self.u = u
     
        sv_ind= (u>1e-4).flatten()
        sv_ind_leqC=(u<C).flatten()
        sv_ind_mh=np.logical_and(sv_ind, sv_ind_leqC)
        
        self.b = np.mean(trainlabels[sv_ind_mh] - (u*trainlabels).T @ K @ (np.eye(m)[sv_ind_mh]).T)
    
    # Specific implementation for hypothesis value (as we don't have w)
    def predict(self, data):
        suma =0
        for i in range(self.m):
            suma += self.u[i]*self.trainlabels[i]*self.kernel.func(self.traindata[i], data)
        return np.sign(suma+self.b)
    
    def decision_function(self, xy):
        P=np.array([])
        for i in range(xy.shape[0]):
            suma =0
            for j in range(self.m):
                suma += self.u[j]*self.trainlabels[j]*self.kernel.func(self.traindata[i], xy[i])
            P=np.append(P, suma+self.b) 
        return P