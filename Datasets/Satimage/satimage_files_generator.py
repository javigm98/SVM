# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:38:44 2021

@author: javig
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold

def preprocess_file(old_file, new_file):
    data = []
    with open(old_file) as f:
        lines = f.readlines()
    for line in lines:
        line_data = []
        line=line.split(' ')[:-1]
        if(line[0]=='6'):
            label = 1
        else:
            label = -1
        i=1 # feature index
        j=1 # line element index
        while j < len(line):
            while int(line[j].split(':')[0])!=i and i < 37:
                line_data.append(np.nan)
                i+=1
            line_data.append(float(line[j].split(':')[1]))
            i+=1
            j+=1
        line_data.append(label)
        data.append(line_data)   
        
    np.savetxt(new_file, np.array(data))

def merge_train_files():
    df1 = pd.read_csv('train1.txt', delimiter=' ', header=None).dropna()
    print(df1)
    df2 = pd.read_csv('train2.txt', delimiter=' ', header=None).dropna()
    df_combined = pd.concat([df1, df2])
    
    print(df_combined.values)
    np.savetxt('train.txt', df_combined.values)
            
def clean_test_dataset():
    df = pd.read_csv('test1.txt', delimiter=' ', header=None).dropna()
    np.savetxt('test.txt', df.values)
    
def create_fold_files():
    # Create directory if they don't exist
    if not os.path.exists('CrossValidation'):
        os.mkdir('CrossValidation')
        for i in range (1,11):
            if not os.path.exists('CrossValidation/Fold{}'.format(i)):
                os.mkdir('CrossValidation/Fold{}'.format(i))
    kf=KFold(n_splits=10)
    data = np.loadtxt('train.txt')
    i=1
    for train_index, test_index in kf.split(data):
        data_train, data_test = data[train_index], data[test_index]
        np.savetxt('CrossValidation/Fold{}/cv-train.txt'.format(i), data_train)
        np.savetxt('CrossValidation/Fold{}/cv-test.txt'.format(i), data_test)
        i+=1

'''
preprocess_file('satimage.scale.tr.txt', 'train1.txt')
preprocess_file('satimage.scale.val.txt', 'train2.txt')
preprocess_file('satimage.scale.t.txt', 'test1.txt')
merge_train_files()
clean_test_dataset()
'''
create_fold_files()
        
        