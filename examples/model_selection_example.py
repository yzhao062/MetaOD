"""MetaOD prediction with the trained model
"""
# License: BSD 2 clause


# environment setting
import os
import sys
from zipfile import ZipFile
from joblib import load
import numpy as np
import urllib.request 
import pickle

# temporary solution for relative imports in case pyod is not installed
# if metaod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.utils.data import generate_data
from metaod.models.gen_meta_features import generate_meta_features
from metaod.models.utility import prepare_trained_model

from metaod.models.core import MetaODClass
from metaod.models.predict_metaod import select_model

# def prepare_trained_model(url='https://github.com/yzhao062/MetaOD/raw/master/saved_models/trained_models.zip', 
#                           filename='trained_models.zip',
#                           save_path='trained_models'):
            
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
        
#     urllib.request.urlretrieve(url, filename)
    
#     # print(os.path.join(os.path.dirname(os.path.realpath(__file__)),
#     # #                       'trained_models.zip'))
#     # #todo: verify file exists
#     with ZipFile(filename, 'r') as zip:
#         # # printing all the contents of the zip file
#         # zip.printdir()
#         # extracting all the files
#         print('Extracting trained models now...')
#         zip.extractall()
#         print('Finish extracting models')

if __name__ == "__main__":

    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points
    
    # Generate sample data
    X_train, y_train, X_test, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=3,
                      contamination=contamination,
                      random_state=42)
    
    prepare_trained_model()
    clf_setting = select_model(X_train, n_selection=10)
    
    # # # load and unzip models
    # prepare_trained_model()


    # url='https://github.com/yzhao062/MetaOD/raw/master/saved_models/trained_models.zip'
    # filename='trained_models.zip'
    # save_path='trained_models'
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
        
    # urllib.request.urlretrieve(url, filename)
    
    # # print(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    # # #                       'trained_models.zip'))
    # # #todo: verify file exists
    # with ZipFile(filename, 'r') as zip:
    #     # # printing all the contents of the zip file
    #     # zip.printdir()
    #     # extracting all the files
    #     print('Extracting trained models now...')
    #     zip.extractall()
    #     print('Finish extracting models')