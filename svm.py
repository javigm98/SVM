# Javier Guzmán Muñoz

from abc import abstractmethod
import numpy as np
from math import exp, tanh
from cvxopt import matrix, solvers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt

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
        h=matrix(np.hstack([np.zeros(m), np.ones(m)*C]))

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
        self.svm_clf=SVC(kernel='linear', C=1)
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
        h=matrix(np.hstack([np.zeros(m), np.ones(m)*C]))

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

# Returns the percentage of points missclasified, comparing the values of the
# labels predicted by the algorithm (predictedlabels) and the real ones (testlabels)
def classification_error(predictedlabels, testlabels): 
    
    err = float(np.sum(predictedlabels != testlabels))
    percentage_error = err*100/len(testlabels)
    
    return percentage_error

# Plot data (train and test), solution hyperlane and marginal hyperplanes when
# the dataset is 2-dimensional   
def plot_solution_2d(traindata, testdata, trainlabels, testlabels, predictor):
    # Plot train points (with o)
    plt.scatter(traindata[:, 0], traindata[:, 1], c=trainlabels, s=20, cmap='winter')
    # Plot test points (with x)
    plt.scatter(testdata[:,0], testdata[:,1], c=testlabels, s=20, marker ='x', cmap='winter')
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create a grid of 900 points
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Get the value of the decision function for each point in the grid
    P = predictor.decision_function(xy).reshape(X.shape)
    
    # Plot hyperplanes
    ax.contour(X, Y, P, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    

# Read a txt file containing a dataset, wich each row representing an input data.
# All columns except the last one contain the value of the point's features and the
# las one is its label.    
def read_data_txt(file):
    data = np.loadtxt(file)
    valuedata=data[:,:-1]
    labelsdata=np.array([data[:,-1]]).T

    return valuedata, labelsdata

# n-fold cross validation algorithm to determine the optimal C value for the algorithm
#   -C_values: values of parameter C to be tested
#   -files: list of lists. Each element represents a fold step and contains two
#       strings with the train and test files for that fold.
#   -n: number of folds
#   -predClass: predictor class to train and test
#   -arg: additional arguments to use when creating the predictor. We'll use this
#       when using predictors with kernels
def n_fold_cross_validation(C_values, files, n, predClass, arg = None):
    errors_c=[]
    for c in C_values:
        errors=[]
        for i in range(n):
            if arg is None:
                svm = predClass()
            else:
                svm = predClass(arg)
            traindata, trainlabels = read_data_txt(files[i][0])
            svm.train(traindata, trainlabels, c)
            testdata, testlabels= read_data_txt(files[i][1])
            testpredictions = svm.predict_multiple(testdata)
            errors.append(classification_error(testpredictions, testlabels))
        errors_c.append(sum(errors)/len(errors))
    return C_values[np.argmin(errors_c)]


                

def spam_dataset(predictor, kernel=None):
    C_values=[1,10, 1e2, 1e3, 1e4]

    files=[]
    validation_files_root='/Datasets/Spambase/CrossValidation'
    for i in range(5):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C = n_fold_cross_validation(C_values, files, 5, predictor, kernel)       # Put values of C here   
    print('Optimal C value: ', C)
    
    if kernel is None:
        svm = predictor()
    else:
        svm = predictor(kernel)
    traindata, trainlabels = read_data_txt('Datasets/Spambase/train.txt')
    svm.train(traindata, trainlabels, C)
    testdata, testlabels = read_data_txt('Datasets/Spambase/test.txt')
    predictedlabels = svm.predict_multiple(testdata)

    error= classification_error(predictedlabels, testlabels)
    print('Error: ', error)
    
def wines_dataset(predictor, kernel=None):
    C_values=[1,10, 1e2, 1e3, 1e4]

    files=[]
    validation_files_root='Datasets/Wines/CrossValidation'
    for i in range(5):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C = n_fold_cross_validation(C_values, files, 5, predictor, kernel)       # Put values of C here   
    print('Optimal C value: ', C)
    
    if kernel is None:
        svm = predictor()
    else:
        svm = predictor(kernel)
    traindata, trainlabels = read_data_txt('Datasets/Wines/train.txt')
    svm.train(traindata, trainlabels, C)
    testdata, testlabels = read_data_txt('Datasets/Wines/test.txt')
    predictedlabels = svm.predict_multiple(testdata)

    error= classification_error(predictedlabels, testlabels)
    print('Error: ', error)
    
def synthetic_dataset(predictor, kernel=None):
    C_values=[1,10,1e2,1e3,1e4]
    files=[]
    validation_files_root='Datasets/Synthetic/CrossValidation'
    for i in range(5):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C = n_fold_cross_validation(C_values, files, 5, predictor, kernel)       # Put values of C here   
    print('Optimal C value: ', C)
    
    if kernel is None:
        svm = predictor()
    else:
        svm = predictor(kernel)
    traindata, trainlabels = read_data_txt('Datasets/Synthetic/train.txt')
    svm.train(traindata, trainlabels, C)
    testdata, testlabels = read_data_txt('Datasets/Synthetic/test.txt')
    predictedlabels = svm.predict_multiple(testdata)
    
    plot_solution_2d(traindata, testdata, trainlabels, testlabels, svm)
    error= classification_error(predictedlabels, testlabels)
    print('Error: ', error)
    

def iris_2d_dataset(predictor, kernel=None):
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)] # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
    y = np.array([y])
    y = y.T
    y = 2*y-1

    traindata, testdata, trainlabels, testlabels= train_test_split(X,y, test_size=0.33, random_state=42)
    
    if kernel is None:
        svm=predictor()
    else:
        svm = predictor(kernel)

    svm.train(traindata, trainlabels, 1)

    predictedlabels = svm.predict_multiple(testdata)
    
    plot_solution_2d(traindata, testdata, trainlabels, testlabels, svm)
    error= classification_error(predictedlabels, testlabels)
    print('Error: ', error)

def main():
    wines_dataset(SVM_predictor_primal)

if __name__== '__main__':
    main()