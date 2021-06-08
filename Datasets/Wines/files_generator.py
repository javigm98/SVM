# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:56:18 2021

@author: javig
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import os


def create_txt_file_wines():
    df_red=pd.read_csv('winequality-red.csv', delimiter=";")
    df_red = df_red.drop(columns=['quality'])
    df_red['Class']=pd.Series([1 for x in range(len(df_red.index))])
    
    df_white=pd.read_csv('winequality-white.csv', delimiter=";")
    df_white = df_white.drop(columns=['quality'])
    df_white['Class']=pd.Series([-1 for x in range(len(df_white.index))])
    
    df_combined = pd.concat([df_white, df_red])
    
    np.savetxt('dataset_wines.txt', df_combined.values, fmt='%d')

    
def create_train_test_files():
    data = np.loadtxt('dataset_wines.txt')
    traindata, testdata = train_test_split(data, test_size=0.9, random_state=42)
    
    np.savetxt('train.txt', traindata)
    np.savetxt('test.txt', testdata)
    

def create_fold_files():
    # Create directory if they don't exist
    if not os.path.exists('CrossValidation'):
        os.mkdir('CrossValidation')
        for i in range (1,6):
            if not os.path.exists('CrossValidation/Fold{}'.format(i)):
                os.mkdir('CrossValidation/Fold{}'.format(i))
    kf=KFold(n_splits=5)
    data = np.loadtxt('train.txt')
    i=1
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        np.savetxt('CrossValidation/Fold{}/cv-train.txt'.format(i), data_train)
        np.savetxt('CrossValidation/Fold{}/cv-test.txt'.format(i), data_test)
        i+=1
    
create_train_test_files()    
create_fold_files()
    
    
    
    
    
    
    