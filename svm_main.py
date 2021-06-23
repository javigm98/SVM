# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:55:02 2021

@author: javig
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import kernels
import svm_predictors

import matplotlib.pyplot as plt
import time

# Returns the fraction of points missclasified, comparing the values of the
# labels predicted by the algorithm (predictedlabels) and the real ones (testlabels)
def zero_one_loss(predictedlabels, testlabels): 
    
    err = float(np.sum(predictedlabels != testlabels))
    percentage_error = err/len(testlabels)
    
    return percentage_error

# Returns the value for the empirical margin loss for a set of predicted labels
# and their real values.
def empirical_margin_loss(predictedlabels, testlabels, rho=0.7):
    predictedlabels = predictedlabels.ravel()
    testlabels=testlabels.ravel()
    
    phi_values=np.array([])
    for i in range(len(testlabels)):
        phi_values=np.append(phi_values, min(1,max(0,1-((predictedlabels[i]*testlabels[i])/rho))))
    
    return np.sum(phi_values)/len(testlabels)

# Evaluate a set of test points, get their classification labels and calculate zero-one error associated
def evaluate_and_error_zero_one(testdata, testlabels, model):
    testpredictions = model.predict_multiple(testdata)
    return zero_one_loss(testpredictions, testlabels)

# Evaluate a set of test points, get the value of the decision function for each of them and calculate
# empirical margin loss associated    
def evaluate_and_error_margin(testdata, testlabels, model, rho=0.7):
    testpredictions = model.decision_function(testdata)
    return empirical_margin_loss(testpredictions, testlabels, rho)

# Plot data (train and test), solution hyperlane and marginal hyperplanes when
# the dataset is 2-dimensional   
def plot_solution_2d(traindata, testdata, trainlabels, testlabels, predictor, save_dir, title):
    plt.close('all')
    # Plot train points (with o)
    plt.scatter(traindata[:, 0], traindata[:, 1], c=trainlabels, s=20, cmap='winter', label='train data')
    # Plot test points (with x)
    plt.scatter(testdata[:,0], testdata[:,1], c=testlabels, s=20, marker ='x', cmap='winter', label='test dada')
    plt.legend()
    
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
    plt.title(title)
    plt.savefig(save_dir)
    

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
def n_fold_cross_validation(C_values, files, n, predClass, arg = None, loss='zero-one', rho=0.7):
    errors_c=[]
    t0 = time.time()
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
            if loss == 'zero-one':
                errors.append(evaluate_and_error_zero_one(testdata, testlabels, svm))
            elif loss == 'margin':
                errors.append(evaluate_and_error_margin(testdata, testlabels, svm, rho))
        errors_c.append(sum(errors)/len(errors))
    t1 = time.time()
    print("\n-----------------------------------------")
    print("Errors for each C value: ", errors_c)
    print("CV time (s): ", t1-t0)
    print("-----------------------------------------\n")
    
    return C_values[np.argmin(errors_c)]

def n_fold_cross_validation_pol_kernel(C_values, d_values, files, n, loss='zero-one', rho=0.7):
    errors_c=[]
    t0=time.time()
    for c in C_values:
        errors_d=[]
        for d in d_values:
            errors=[]
            kernel = kernels.Polynomial_kernel(c=1, d=d)
            for i in range(n):
                svm = svm_predictors.SVM_predictor_kernel(kernel)
                traindata, trainlabels= read_data_txt(files[i][0])
                svm.train(traindata, trainlabels, c)
                testdata, testlabels = read_data_txt(files[i][1])
                if loss == 'zero-one':
                    errors.append(evaluate_and_error_zero_one(testdata, testlabels, svm))
                elif loss == 'margin':
                    errors.append(evaluate_and_error_margin(testdata, testlabels, svm, rho))
            errors_d.append(sum(errors)/len(errors))
        errors_c.append(errors_d)
    t1 = time.time()
    print("\n-----------------------------------------")
    print("Errors for each C and d value: ", errors_c)
    print("CV time (s): ", t1-t0)
    print("-----------------------------------------\n")
    
    ind_i=0
    ind_j=0
    min_err = np.infty
    
    for i in range(0, len(errors_c)):
        for j in range(0, len(errors_c[i])):
            if errors_c[i][j] < min_err:
                min_err= errors_c[i][j]
                ind_i=i
                ind_j=j
    return C_values[ind_i], d_values[ind_j]


def run_dataset_with_cv(name, predictor, kernel, n=5, loss='zero-one', rho=0.7):
    C_values=[1,10, 1e2, 1e3, 1e4]

    files=[]
    validation_files_root='Datasets/{}/CrossValidation'.format(name)
    for i in range(n):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C = n_fold_cross_validation(C_values, files, n, predictor, kernel, loss, rho)       # Put values of C here   
    print("\n-----------------------------------------")
    print('Optimal C value: ', C)
    
    if kernel is None:
        svm = predictor()
    else:
        svm = predictor(kernel)
    traindata, trainlabels = read_data_txt('Datasets/{}/train.txt'.format(name))
    t0=time.time()
    svm.train(traindata, trainlabels, C)
    t1=time.time()
    testdata, testlabels = read_data_txt('Datasets/{}/test.txt'.format(name))
    
    if loss=='zero-one':
        error= evaluate_and_error_zero_one(testdata, testlabels, svm)
    else:
        error = evaluate_and_error_margin(testdata, testlabels, svm, rho)
        
    print('Error: ', error)
    print('Train time(s): ', t1-t0)
    print("-----------------------------------------\n")

    if(traindata.shape[1]==2 and predictor != svm_predictors.SVM_predictor_kernel):
        pred_save_name, pred_save_title = get_graph_name_predictor(predictor)
        save_name = 'Graphs/{}/{}_C_{}.png'.format(name, pred_save_name, C)
        title = pred_save_title + ' for ' + name + ' dataset with C=' + str(C)
        plot_solution_2d(traindata, testdata, trainlabels, testlabels, svm, save_name, title)

def run_dataset_with_cv_and_pol_kernel(name, n=5, loss='zero-one', rho=0.7):
    C_values=[1,10, 1e2, 1e3, 1e4]
    d_values=[1,2,3,4]

    files=[]
    validation_files_root='Datasets/{}/CrossValidation'.format(name)
    for i in range(n):
        files.append([validation_files_root+'/Fold{}/cv-train.txt'.format(i+1), validation_files_root+'/Fold{}/cv-test.txt'.format(i+1)])

    C,d = n_fold_cross_validation_pol_kernel(C_values, d_values, files, n, loss, rho)       # Put values of C here   
    print("\n-----------------------------------------")
    print('Optimal C value: ', C)
    print('Optimal d value: ', d)
    
    kernel = kernels.Polynomial_kernel(c=1, d=d)
    svm=svm_predictors.SVM_predictor_kernel(kernel)
    traindata, trainlabels = read_data_txt('Datasets/{}/train.txt'.format(name))
    t0=time.time()
    svm.train(traindata, trainlabels, C)
    t1=time.time()
    testdata, testlabels = read_data_txt('Datasets/{}/test.txt'.format(name))
    
    if loss=='zero-one':
        error= evaluate_and_error_zero_one(testdata, testlabels, svm)
    else:
        error = evaluate_and_error_margin(testdata, testlabels, svm, rho)
        
    print('Error: ', error)
    print('Train time(s): ', t1-t0)
    print("-----------------------------------------\n")

def get_graph_name_predictor(predictor):
    if predictor == svm_predictors.SVM_predictor_primal:
        return 'primal_predcitor', 'SVM primal predictor'
    elif predictor==svm_predictors.SVM_predictor_dual:
        return 'dual_predictor', 'SVM dual predictor'
    elif predictor == svm_predictors.SVM_predictor_sklearn:
        return 'sklearn_predictor', 'SVM sklearn predictor'
    else:
        return 'primal_predictor_separable', 'Primal predictor separable'
    
def iris_2d_dataset(predictor, kernel=None, loss='zero-one', rho=0.7):
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)] # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
    y = np.array([y])
    y = y.T
    y = 2*y-1

    traindata, testdata, trainlabels, testlabels= train_test_split(X,y, test_size=0.6, random_state=42)
    
    if kernel is None:
        svm=predictor()
    else:
        svm = predictor(kernel)
    
    t0=time.time()
    svm.train(traindata, trainlabels, 1)
    t1=time.time()

    
    if loss=='zero-one':
        error= evaluate_and_error_zero_one(testdata, testlabels, svm)
    else:
        error=evaluate_and_error_margin(testdata, testlabels, svm, rho)
        
    print("\n-----------------------------------------")
    print('Error: ', error)
    print('Train time (s): ', t1-t0)
    print("-----------------------------------------\n")
    
    if predictor != svm_predictors.SVM_predictor_kernel:
        pred_save_name, pred_save_title = get_graph_name_predictor(predictor)
        save_name = 'Graphs/Iris_2d/{}.png'.format(pred_save_name)
        title = pred_save_title + ' for Iris 2d dataset C=1'
        plot_solution_2d(traindata, testdata, trainlabels, testlabels, svm, save_name, title)
        
def separable_dataset(predictor, kernel=None, loss='zero-one', rho=0.7):
    initial_centers = [[1, -1], [-1, 1]]
    X,y = make_blobs(n_samples=1000, centers=initial_centers, 
                                cluster_std=0.6,random_state=0)
    y = np.array([y])
    y = y.T
    y = 2*y-1
    traindata, testdata, trainlabels, testlabels= train_test_split(X,y, test_size=0.6, random_state=42)
    if kernel is None:
        svm=predictor()
    else:
        svm = predictor(kernel)
    
    if predictor != svm_predictors.SVM_predictor_primal_separable:
        t0=time.time()
        svm.train(traindata, trainlabels, 1)
        t1=time.time()
    else:
        t0=time.time()
        svm.train(traindata, trainlabels)
        t1=time.time()

    
    if loss=='zero-one':
        error= evaluate_and_error_zero_one(testdata, testlabels, svm)
    else:
        error=evaluate_and_error_margin(testdata, testlabels, svm, rho)
        
    print("\n-----------------------------------------")
    print('Error: ', error)
    print('Train time (s): ', t1-t0)
    print("-----------------------------------------\n")
    
    if predictor != svm_predictors.SVM_predictor_kernel:
        pred_save_name, pred_save_title = get_graph_name_predictor(predictor)
        save_name = 'Graphs/Separable/{}.png'.format(pred_save_name)
        title = pred_save_title + ' for separable dataset'
        plot_solution_2d(traindata, testdata, trainlabels, testlabels, svm, save_name, title)       
    
    
def enter_option(value_min, value_max, text):
    while True:
        try:
            opt = int(input(text))
        except ValueError:
            print("Input must be a integer name!")
            continue
        else:
            if value_max==np.inf and opt>= value_min or opt in range(value_min,value_max+1):
                break
            else:
                print("Input must be between {} and {}".format(value_min, value_max))
                
    return opt

def enter_positive_float(text):
    while True:
        try:
            value = float(input(text))
        except ValueError:
            print("Input must be a float number!")
        
        else:
            if value <=0:
                print("Value must be > 0!")
            else:
                break
    return value

def enter_positive_or_zero_float(text):
    while True:
        try:
            value = float(input(text))
        except ValueError:
            print("Input must be a float number!")
        
        else:
            if value <0:
                print("Value must be >= 0!")
            else:
                break
    return value

def main():
    while True:
        cv_kernel_opt=0
        
        print("Choose the dataset you want to work with:")
        print("1. Spam Messages Dataset (57 features, 250 test samples, 4351 train samples)")
        print("2. Portuguese Wines Dataset (11 features, 649 train samples, 5848 test samples)")
        print("3. Iris Dataset 2D (2 features, 60 train samples, 90 test samples)")
        print("4. Synthetic Dataset (2 features, 250 train samples, 750 test samples)")
        print("5. Satimages Dataset (38 features, 3188 train samples, 1431 test samples)")
        print("6. Separable Dataset (2 features, 400 train samples, 600 test samples)")
        print("7. Exit")
        
        dataset_opt=enter_option(1,7, "Enter an option: ")
        
        if dataset_opt==7:
            break
        
        print("Choose the predictor SVM predictor to use:")
        print("1. SVM predictor solving the primal problem")
        print("2. SVM predictor solving the dual problem")
        print("3. SVM predictor solving the dual problem and using a kernel")
        print("4. SVM predictor implemented in sklearn ")
        if dataset_opt==6:
            print("5. SVM predictor solving the primal problem in the separable case")
            predictor_opt=enter_option(1,5, "Enter an option: ")
        
        else:
            predictor_opt=enter_option(1,4, "Enter an option: ")
        kernel = None
        
        print("Choose the loss function to consider:")
        print("1. Zero-one loss")
        print("2. Empirical margin loss")
        
        loss_opt=enter_option(1,2, "Enter an option: ")
        rho=0.7
        loss='zero-one'
        if loss_opt==2:
            rho=enter_positive_float("Enter the rho value to consider (default 0.7): ")
            loss='margin'
        
        if predictor_opt==1:
            predictor = svm_predictors.SVM_predictor_primal
            
        elif predictor_opt==2:
            predictor = svm_predictors.SVM_predictor_dual
            
        elif predictor_opt==3:
            predictor = svm_predictors.SVM_predictor_kernel
            print("Enter the kernel you want to use:")
            print("1. Polynomial kernel (x*x'+c)^d")
            print("2. Gaussian kernel exp((-||x-x'||^2)/(2sigma^2)")
            print("3. Sigmoid kernel tanh(a(x*x')+b")
            kernel_opt = enter_option(1,3, "Enter an option: ")
            if kernel_opt==1 and dataset_opt != 3 and dataset_opt != 6:
                print("How do you want to choose the c and d values?")
                print("1. Enter the exact c and d values")
                print("2. Fix c=1 and select d via cross validation")
                cv_kernel_opt= enter_option(1,2, 'Enter an option: ')
                if cv_kernel_opt==1:
                    c = enter_positive_float("Enter c value: ")
                    d = enter_option(1, np.infty, "Enter d value: ")
                    kernel = kernels.Polynomial_kernel(c,d)
            elif kernel_opt==1:
                c = enter_positive_float("Enter c value: ")
                d = enter_option(1, np.infty, "Enter d value: ")
                kernel = kernels.Polynomial_kernel(c,d)
            elif kernel_opt==2:
                sigma=enter_positive_float("Enter sigma value: ")
                kernel= kernels.Gaussian_kernel(sigma)
            else:
                a = enter_positive_or_zero_float("Enter a value: ")
                b = enter_positive_or_zero_float("Enter b value: ")
                kernel = kernels.Sigmoid_kernel(a, b)
        
        elif predictor_opt ==4:
            predictor = svm_predictors.SVM_predictor_sklearn
        else:
            predictor =svm_predictors.SVM_predictor_primal_separable
        
        if dataset_opt == 1:
            if cv_kernel_opt==2:
                run_dataset_with_cv_and_pol_kernel('Spambase', loss=loss, rho=rho)
            else:
                run_dataset_with_cv('Spambase', predictor, kernel, loss=loss, rho=rho)
        elif dataset_opt==2:
            if cv_kernel_opt==2:
                run_dataset_with_cv_and_pol_kernel('Wines', loss=loss, rho=rho)
            else:
                run_dataset_with_cv('Wines', predictor, kernel, loss=loss, rho=rho)
        elif dataset_opt==3:
            iris_2d_dataset(predictor, kernel, loss=loss, rho=rho)
        elif dataset_opt==4:
            if cv_kernel_opt==2:
                run_dataset_with_cv_and_pol_kernel('Synthetic', loss=loss, rho=rho)
            else:
                run_dataset_with_cv('Synthetic', predictor, kernel, loss=loss, rho=rho)
        elif dataset_opt==5:
            if cv_kernel_opt==2:
                run_dataset_with_cv_and_pol_kernel('Satimage', loss=loss, rho=rho)
            else:
                run_dataset_with_cv('Satimage', predictor, kernel, n=10, loss=loss, rho=rho)
        elif dataset_opt==6:
            separable_dataset(predictor, kernel, loss=loss, rho=rho)
        else:
            break

if __name__== '__main__':
    main()