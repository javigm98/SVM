from abc import abstractmethod
import numpy as np
from math import exp, tanh
from cvxopt import matrix, solvers
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt


class SVM_predictor():
    def __init__(self):
        self.b=0
        self.w = None
    @abstractmethod
    def train(self, train_data, trainlabels, C):
        raise NotImplementedError

    def predict(self, data):
        return np.sign(np.matmul(self.w.T, data.T)+self.b)

    def predict_multiple(self, data):
        predictions= np.array([])
        for x in data:
            predictions = np.append(predictions, self.predict(x))
        return np.array([predictions]).T
    
    def decision_fnction(self, xy):
        P=np.array([])
        for i in range(xy.shape[0]):
           P=np.append(P,np.dot(xy[i], self.w)+self.b) 
        return P




class SVM_predictor_primal(SVM_predictor):
    def __init__(self):
        super().__init__()

    def train(self,traindata, trainlabels, C):
    
#     INPUT : 
#     traindata   - m X n matrix, where m is the number of training points
#     trainlabels - m X 1 vector of training labels for the training data
#     C           - SVM regularization parameter (positive real number)
#     
#     
#     OUTPUT :
#     returns the structure 'model' which has the following fields:
#     
#     b - SVM bias term
#     sv - the subset of training data, which are the support vectors
#     sv_alphas - m X 1 vector of support vector coefficients
#     sv_labels - corresponding labels of the support vectors

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
        #self.xi=np.array(solution['x'][n+1:len(solution['x'])])

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
        '''
        sv_x = traindata[sv_ind]
        sv_y = trainlabels[sv_ind]
        sv_u = trainlabels[sv_ind]
        '''

        sv_ind_leqC=(u<C).flatten()

        sv_ind_mh=np.logical_and(sv_ind, sv_ind_leqC)

        self.w = ((trainlabels * u).T @ traindata).reshape(-1,1)

        self.b = np.mean(trainlabels[sv_ind_mh] - np.dot(traindata[sv_ind_mh], self.w))

class SVM_predictor_sklearn(SVM_predictor):
    def __init__(self):
        super().__init__()
    
    def train(self, traindata, trainlabels, C):
        self.svm_clf=SVC(kernel='linear', C=1)
        trainlabels = np.ravel(trainlabels)
        self.svm_clf.fit(traindata, trainlabels)
        
    def predict_multiple(self, data):
        return np.array([self.svm_clf.predict(data)]).T
    
    def decision_function(self, xy):
        return self.svm_clf.decision_function(xy)

class Kernel():
    def __init__(self):
        pass
    @abstractmethod
    def func(self, x, y):
        raise NotImplementedError

class SVM_predictor_kernel(SVM_predictor):
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

        #self.w = ((trainlabels * u).T @ traindata).reshape(-1,1)

        self.b = np.mean(trainlabels[sv_ind_mh] - (u*trainlabels).T @ K @ (np.eye(m)[sv_ind_mh]).T)

    def predict(self, data):
        suma =0
        for i in range(self.m):
            suma += self.u[i]*self.trainlabels[i]*self.kernel.func(self.traindata[i], data)
        return np.sign(suma+self.b)


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
  
def classification_error(predictedlabels, testlabels):
    


#      This function computes the classification error for the predicted labels
#       with respect to the ground truth. The returned error value is a real 
#       number between 0 and 1 (fraction of misclassications).
# 
#   testlabels: vector of true labels (each label +1/-1)
#   predictedlabels: vector of predicted labels (each prediction +1/-1)
#   percentage_error: classification error (percentage of misclassifications)

#   You don't need to write anyhing here
    
    
    err = float(np.sum(predictedlabels != testlabels))
    percentage_error = err*100/len(testlabels)
    
    return percentage_error
    
def plot_solution_2d(traindata, testdata, trainlabels, testlabels, predictor):
    plt.scatter(traindata[:, 0], traindata[:, 1], c=trainlabels, s=20, cmap='winter')
    plt.scatter(testdata[:,0], testdata[:,1], c=testlabels, s=20, marker ='x', cmap='winter')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = predictor.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    
def read_data_txt(file):
    data = np.loadtxt(file)
    traindata=data[:,:-1]
    trainlabels=np.array([data[:,-1]]).T

    return traindata, trainlabels



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
    validation_files_root='Programming assignments/Machine-Learning-master/Machine-Learning-master/Datasets/2018/Spambase/CrossValidation'
    for i in range(5):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C = n_fold_cross_validation(C_values, files, 5, predictor, kernel)       # Put values of C here   
    print('Optimal C value: ', C)
    
    if kernel is None:
        svm = predictor()
    else:
        svm = predictor(kernel)
    traindata, trainlabels = read_data_txt('Programming assignments/Machine-Learning-master/Machine-Learning-master/Datasets/2018/Spambase/train.txt')
    svm.train(traindata, trainlabels, C)
    testdata, testlabels = read_data_txt('Programming assignments/Machine-Learning-master/Machine-Learning-master/Datasets/2018/Spambase/test.txt')
    predictedlabels = svm.predict_multiple(testdata)

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
    iris_2d_dataset(SVM_predictor_sklearn)



    #    Now call the above defined functions to learn the SVM model using training 
    #    data, use the learnt model to classify the test data and finally find the 
    #    classification error of your learnt model. Store the error in a variable 
    #    named ERROR. 

    #    Write your code here to find ERROR





    #==============================================================================
if __name__== '__main__':
    main()