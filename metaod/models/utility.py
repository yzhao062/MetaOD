# -*- coding: utf-8 -*-
import os
import numpy as np
import arff
from zipfile import ZipFile
import urllib.request 

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

def argmaxatn(w, nth):
    w = np.asarray(w).ravel()
    t = np.argsort(w)
    return t[-1*nth]

def fix_nan(X):
    col_mean = np.nanmean(X, axis = 0) 
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1]) 
    
    return X


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes

def prepare_trained_model(url='https://github.com/yzhao062/MetaOD/blob/master/saved_models/metaod_models.zip', filename='metaod_models.zip'):
    urllib.request.urlretrieve(url, filename)
    #todo: verify file exists
    with ZipFile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'trained_models.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()
    
        # extracting all the files
        print('Extracting trained models now...')
        zip.extractall(path='trained_models')
        print('Finish extracting models')
    
    