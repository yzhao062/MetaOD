"""MetaOD prediction with the trained model
"""
# License: BSD 2 clause


# environment setting
from zipfile import ZipFile
import os
from joblib import load
from pyod.utils.data import generate_data
import numpy as np

from metaod.models.gen_meta_features import generate_meta_features
from pyod.utils.data import generate_data

def get_top_models(p, n):
    return np.flip(np.argsort(p))[:n]

def select_model(X, trained_model_location="trained_models", n_selection=1):
    
    # print(os.path.realpath(__file__))
    # unzip trained models
    # with ZipFile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                           'trained_models.zip'), 'r') as zip:
    #     # # printing all the contents of the zip file
    #     # zip.printdir()
    
    #     # extracting all the files
    #     print('Extracting trained models now...')
    #     zip.extractall(path='trained_models')
    #     print('Finish extracting models')

    # load PCA scalar
    meta_scalar = load(os.path.join(trained_model_location,"meta_scalar.joblib"))
    # generate meta features         
    meta_X, _ = generate_meta_features(X)
    meta_X = meta_scalar.transform(np.asarray(meta_X).reshape(1, -1))
    
    # use all trained models for ensemble
    trained_models = [
        "train_0.joblib", 
        # "train_1.joblib", 
        # "train_42.joblib"
        ]
    print(os.getcwd())
    # # load trained models
    model_lists = list(load(os.path.join(trained_model_location,"model_list.joblib")))
    
    predict_scores = np.zeros([len(trained_models), len(model_lists)])
    
    for i, model in enumerate(trained_models):
        clf = load(os.path.join(trained_model_location, model))
        # w = load (model)
        predict_scores[i,] = clf.predict(meta_X)
        predicted_scores_max = np.nanargmax(predict_scores[i,])
        # print('top model', model_lists[predicted_scores_max])
    combined_predict = np.average(predict_scores, axis=0)
    
    predicted_scores_sorted = get_top_models(combined_predict, n_selection)
    predicted_scores_max = np.nanargmax(combined_predict)
    
    print('top model', model_lists[predicted_scores_sorted[0]])
    
    return np.asarray(model_lists)[predicted_scores_sorted]

# if __name__ == "__main__":

#     contamination = 0.1  # percentage of outliers
#     n_train = 200  # number of training points
#     n_test = 100  # number of testing points

#     # Generate sample data
#     X_train, y_train, X_test, y_test = \
#         generate_data(n_train=n_train,
#                       n_test=n_test,
#                       n_features=3,
#                       contamination=contamination,
#                       random_state=42)

#     clf_setting = select_model(X_train, n_selection=10)
