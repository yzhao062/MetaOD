# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:27 2020

@author: yuezh
"""

import os
import random
import pandas as pd
import numpy as np

from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat

from utility import read_arff, fix_nan
from joblib import dump

from gen_meta_features import generate_meta_features
from core import MetaODClass

# read in performance table
roc_df = pd.read_excel(os.path.join('data', 'performance_table.xlsx'),
                       sheet_name='AP')

# trim the table
roc_mat = roc_df.to_numpy()
roc_mat_red = fix_nan(roc_mat[2:, 4:].astype('float'))

# get statistics of the training data
n_datasets, n_configs = roc_mat_red.shape[0], roc_mat_red.shape[1]
data_headers = roc_mat[2:, 0]
config_headers = roc_df.columns[4:]
dump(config_headers, 'model_list.joblib')

# %%

# build meta-features
meta_mat = np.zeros([n_datasets, 200])

# read in mat files
mat_file_list = [
    'annthyroid.mat',
    'arrhythmia.mat',
    'breastw.mat',
    'glass.mat',
    'ionosphere.mat',
    'letter.mat',
    'lympho.mat',
    'mammography.mat',
    'mnist.mat',
    'musk.mat',
    'optdigits.mat',
    'pendigits.mat',
    'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    'shuttle.mat',
    'smtp_n.mat',
    'speech.mat',
    'thyroid.mat',
    'vertebral.mat',
    'vowels.mat',
    'wbc.mat',
    'wine.mat',
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',

    'ALOI',  # too large
    'Glass',  # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC',  # too small
    'WDBC',  # too small
    'WPBC',  # too small
]

for j in range(23):
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join("data", "ODDS", mat_file))
    X = mat['X']
    meta_mat[j, :], meta_vec_names = generate_meta_features(X)
    print(j, mat_file)

# read arff files
file_names = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',

    'ALOI',  # too large
    'Glass',  # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC',  # too small
    'WDBC',  # too small
    'WPBC',  # too small
]

#############################################################################
misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                  'KDDCup99']
arff_list = [
    os.path.join('semantic', 'Annthyroid', 'Annthyroid_withoutdupl_07.arff'),
    os.path.join('semantic', 'Arrhythmia', 'Arrhythmia_withoutdupl_46.arff'),
    os.path.join('semantic', 'Cardiotocography',
                 'Cardiotocography_withoutdupl_22.arff'),
    os.path.join('semantic', 'HeartDisease',
                 'HeartDisease_withoutdupl_44.arff'),
    os.path.join('semantic', 'Hepatitis', 'Hepatitis_withoutdupl_16.arff'),
    os.path.join('semantic', 'InternetAds',
                 'InternetAds_withoutdupl_norm_19.arff'),
    os.path.join('semantic', 'PageBlocks', 'PageBlocks_withoutdupl_09.arff'),
    os.path.join('semantic', 'Pima', 'Pima_withoutdupl_35.arff'),
    os.path.join('semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff'),
    os.path.join('semantic', 'Stamps', 'Stamps_withoutdupl_09.arff'),
    os.path.join('semantic', 'Wilt', 'Wilt_withoutdupl_05.arff'),

    os.path.join('literature', 'ALOI', 'ALOI_withoutdupl.arff'),
    os.path.join('literature', 'Glass', 'Glass_withoutdupl_norm.arff'),
    os.path.join('literature', 'PenDigits',
                 'PenDigits_withoutdupl_norm_v01.arff'),
    os.path.join('literature', 'Shuttle', 'Shuttle_withoutdupl_v01.arff'),
    os.path.join('literature', 'Waveform', 'Waveform_withoutdupl_v01.arff'),
    os.path.join('literature', 'WBC', 'WBC_withoutdupl_v01.arff'),
    os.path.join('literature', 'WDBC', 'WDBC_withoutdupl_v01.arff'),
    os.path.join('literature', 'WPBC', 'WPBC_withoutdupl_norm.arff')
]

for j in range(23, 42):
    mat_file = file_names[j - 23]
    mat_file_path = os.path.join("data", "DAMI", arff_list[j - 24])
    X, y, attributes = read_arff(mat_file_path, misplaced_list)
    X = check_array(X).astype('float64')
    meta_mat[j, :], meta_vec_names = generate_meta_features(X)
    print("processing", j, mat_file)

# read emmott dataset
selected_bench = pd.read_csv(os.path.join('data', 'childsets.csv'))[
    'bench.id'].values.tolist()
selected_bench_loc = pd.read_csv(os.path.join('data', 'childsets.csv'))[
    'location'].values.tolist()

for j in range(42, 142):
    print("processing", j, selected_bench_loc[j - 42])
    mat = pd.read_csv(
        os.path.join("data", "Emmott", selected_bench_loc[j - 42]))
    X = mat.to_numpy()[:, 6:].astype(float)
    meta_mat[j, :], meta_vec_names = generate_meta_features(X)

# use cleaned and transformed meta-features
meta_scalar = MinMaxScaler()
meta_mat_transformed = meta_scalar.fit_transform(meta_mat)
meta_mat_transformed = fix_nan(meta_mat_transformed)
dump(meta_scalar, 'meta_scalar.joblib')
# %% train model

# split data into train and valid
seed = 0
full_list = list(range(n_datasets))
random.Random(seed).shuffle(full_list)
n_train = int(0.85 * n_datasets)

train_index = full_list[:n_train]
valid_index = full_list[n_train:]

train_set = roc_mat_red[train_index, :].astype('float64')
valid_set = roc_mat_red[valid_index, :].astype('float64')

train_meta = meta_mat_transformed[train_index, :].astype('float64')
valid_meta = meta_mat_transformed[valid_index, :].astype('float64')

clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=30,
                  learning='sgd')
clf.train(n_iter=50, meta_features=train_meta, valid_meta=valid_meta,
          learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1,
          n_steps=8)

# U = clf.user_vecs
# V = clf.item_vecs

# # # print(EMF.regr_multirf.predict(test_meta).shape)
# predicted_scores = clf.predict(valid_meta)
# predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
# print()
# output transformer (for meta-feature) and the trained clf
dump(clf, 'train_' + str(seed) + '.joblib')

# %%
# import pickle
# MetaODClass.__module__ = "metaod"
# file = open('test.pk', 'wb')
# pickle.dump(clf, file)

# # file = open('rf.pk', 'wb')
# # pickle.dump(clf.user_vecs, file)
