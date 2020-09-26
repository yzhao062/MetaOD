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
        
    clf_setting = select_model(X_train, n_selection=10)
    
    # # load and unzip models
    # prepare_trained_model()

    # # load PCA scalar
    # meta_scalar = load(os.path.join("trained_models","meta_scalar.joblib"))
    # meta_X, _ = generate_meta_features(X_train)
    # meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1))
    
    # model_lists = list(load(os.path.join("trained_models","model_list.joblib")))
    
    # # use all trained models for ensemble
    # trained_models = [
    #     # 'test.pk'
    #     "train_0.joblib", 
    #     # "train_1.joblib", 
    #     # "train_42.joblib"
    #     ]
    
    # for i, model in enumerate(trained_models):
    #     clf = load(os.path.join("trained_models", model))
        
    # # get top 10 models
    # clf_setting = select_model(X_train, n_selection=10)
    # meta_X, _ = generate_meta_features(X_train)
    # meta_X = np.asarray(meta_X).reshape(1, -1)

    # w = load(os.path.join("trained_models", "train_1.joblib"))