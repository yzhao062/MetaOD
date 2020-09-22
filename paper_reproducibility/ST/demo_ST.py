# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
from pyod.utils.utility import argmaxn
import itertools

from scipy.stats import wilcoxon
from numpy.linalg import solve

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_array
from MF_SGD import ExplicitMF
from MF_SGD_Fixed import ExplicitMFF
from MF_SGD_Rank import ExplicitMFRank
from MF_SGD_Rank_Fixed import ExplicitMFRankF
from meta_feature_generation import gen_meta_features
import os

from scipy.sparse.linalg import svds
from scipy.linalg import svd
from numpy import diag

from sklearn.metrics import mean_squared_error
from sklearn.metrics import dcg_score, ndcg_score

from scipy.io import loadmat
import arff


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

    'ALOI', # too large
    'Glass', # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC', # too small
    'WDBC', # too small
    'WPBC', # too small
]


roc_df = pd.read_excel('performance_table.xlsx', sheet_name='AP')

roc_mat = roc_df.to_numpy()
roc_mat_red = roc_mat[2:, 4:-3]

config_headers = roc_df.columns[4:-3]

n_datasets, n_configs = roc_mat_red.shape[0], roc_mat_red.shape[1]
n_datasets = 62 ## cardio removed
data_index = roc_mat[2:, 1].astype(int)
data_headers = roc_mat[2:, 0]


meta_mat = np.zeros([len(mat_file_list)+20, 200])

for j in range(23):
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join("data", "ODDS", mat_file))
    X = mat['X']
    meta_mat[j, :], meta_vec_names = gen_meta_features(X)
    print(j,mat_file )


# Define data file and read X and y
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

    
    mat_file = file_names[j-23]
    mat_file_path = os.path.join("data", "DAMI", arff_list[j-24])
    X, y, attributes = read_arff(mat_file_path, misplaced_list)
    X = check_array(X).astype('float64')
    meta_mat[j, :], meta_vec_names  = gen_meta_features(X)
    print("processing", j, mat_file)
    
    

selected_bench = pd.read_csv('POC_20_Emmott.csv')['bench.id'].values.tolist()
selected_bench_loc = pd.read_csv('POC_20_Emmott.csv')['location'].values.tolist()

mat_file_list.extend(selected_bench)

for j in range(42, 62):
    print("processing", j, selected_bench_loc[j-42])
    mat = pd.read_csv(os.path.join("data", "Emmott", selected_bench_loc[j-42]))
    X = mat.to_numpy()[:, 6:].astype(float)
    meta_mat[j, :], meta_vec_names = gen_meta_features(X)
    
    

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

meta_mat_transformed = MinMaxScaler().fit_transform(meta_mat)
meta_mat_transformed = fix_nan(meta_mat_transformed)

#%%
full_param = []
full_param_headers = []

param_mat = np.zeros([len(config_headers), 14]).astype('object')
param_tracker = 0

classifier_name = "LODA"
param_list = []
param_list_1 = [5, 10, 15, 20, 25, 30]
param_list_2 = [10, 20, 30, 40, 50, 75, 100, 150, 200]

for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 0] = r[0]
    param_mat[param_tracker, 1] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))

classifier_name = "ABOD"
# param_list = [3, 5, 10, 15, 20, 25, 50]
param_list = [3, 5, 10, 15, 20, 25, 50, 60, 70, 80, 90, 100]
for r in param_list:
    param_mat[param_tracker, 2] = r
    param_tracker+=1
    
full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "IForest"
param_list = []
param_list_1 = [10, 20, 30, 40, 50, 75, 100, 150, 200]
param_list_2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 3] = r[0]
    param_mat[param_tracker, 4] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "kNN"
param_list = []
param_list_1 = [1, 5 ,10, 15, 20, 25, 50, 60, 70, 80, 90, 100]
param_list_2 = ['largest', 'mean', 'median']
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 5] = r[0]
    param_mat[param_tracker, 6] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "LOF"
param_list = []
param_list_1 = [1, 5 ,10, 15, 20, 25, 50, 60, 70, 80, 90, 100]
param_list_2 = ['manhattan', 'euclidean', 'minkowski']
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 7] = r[0]
    param_mat[param_tracker, 8] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "HBOS"
param_list = []
param_list_1 = [5, 10, 20, 30, 40, 50, 75, 100]
param_list_2 = [0.1, 0.2, 0.3, 0.4, 0.5]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 9] = r[0]
    param_mat[param_tracker, 10] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "OCSVM"
param_list = []
param_list_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_list_2 = ["linear", "poly", "rbf", "sigmoid"]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_mat[param_tracker, 11] = r[0]
    param_mat[param_tracker, 12] = r[1]
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))



classifier_name = "COF"
param_list = [3, 5, 10, 15, 20, 25, 50]
for r in param_list:
    param_mat[param_tracker, 13] = r
    param_tracker+=1
full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))

a = pd.get_dummies(param_mat[:, 6]).values[:, 1:]
b = pd.get_dummies(param_mat[:, 8]).values[:, 1:]
c = pd.get_dummies(param_mat[:, 12]).values[:, 1:]

param_mat_full = np.concatenate([param_mat, a, b, c], axis=1)
param_mat_full[:, 6] = 0
param_mat_full[:, 8] = 0
param_mat_full[:, 12] = 0

param_mat_full = param_mat_full.astype(float)

# build X, y
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

X_reg = np.zeros([n_datasets*n_configs, int(200+24)])
# X_reg = np.zeros([n_datasets*n_configs, int(200+24)])
y_reg = np.zeros([n_datasets*n_configs,])

for i in range (n_datasets):
    for j in range(n_configs):
        # X_reg[i*n_configs+j, :200] = meta_mat[i, :]
        # X_reg[i*n_configs+j, 200:] = param_mat_full[j, :]
        
        X_reg[i*n_configs+j, :200] = meta_mat[i, :]
        X_reg[i*n_configs+j, 200:] = param_mat_full[j, :]
        y_reg[i*n_configs+j] = roc_mat_red[i, j]
        
X_reg_fixed = np.nan_to_num(X_reg).astype(np.float64)
y_reg_fixed = np.nan_to_num(y_reg).astype(np.float64)

#%%
# all_data_index = np.arange(n_sub_datasets)
all_data_index = list(range(n_datasets))
random_state = np.random.RandomState(42)
random_state.shuffle(all_data_index)

n_folds = 62
# n_datasets_fold = int(n_sub_datasets/n_folds)
n_datasets_fold = int(n_datasets/n_folds)
fold_index_list = []

for i in range(n_folds):
    if i == n_folds-1:
        fold_index_list.append(all_data_index[i*n_datasets_fold:])
    else:
        fold_index_list.append(all_data_index[i*n_datasets_fold: (i+1)*n_datasets_fold])

data_headers_random = []    
for i in fold_index_list:
    data_headers_random.append(data_headers[i][0])
    

#%% This includes additional baselines
from MF_SGD import ExplicitMF
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
# test_roc = []
iForest1 = 133  # Iforest (200, 0.1)
iForest2 = 128  # Iforest (150, 0.5)
LOF1 = 192  # LOF (20, minkowski)
n_selection = 3
headers = ['iForest (200, 0.1)', 'iForest (150, 0.5)', 
           '(LOF (20, minkowski)', 'GB', 
           'EUB', '1st Per Data', '2nd Per Data', '3rd Per Data', '5th Per Data', '10th Per Data', 
           'ALORS_MSE', 'ISAC', 'AS', 'MetaOD_C', 'SS', 'ALORS_F',  
           'ALORS_Rank', 'ALORS_Rank_F', 'ALORS_F_3', 'ALORS_Rank_3', 'ALORS_Rank_F_3',  'MetaOD', 'MetaOD_3']
all_perm = list(itertools.combinations(list(range(23)), 2))



header_cv = np.empty([n_datasets, 23], dtype="object")
roc_cv = np.zeros([n_datasets, 23])
train_dataset_headers = []
test_dataset_headers = []

for ind, data_ind in enumerate(fold_index_list):
    print(ind, data_ind)
    train_index = Diff(all_data_index, data_ind)
    test_index = data_ind
    
    train_meta = meta_mat[train_index, :].astype('float64')
    train_meta[np.isnan(train_meta)] = 0
    
    test_meta =  meta_mat[test_index, :].astype('float64')
    test_meta[np.isnan(test_meta)] = 0
    
    train_meta_transformed = meta_mat_transformed[train_index, :].astype('float64')
    test_meta_transformed =  meta_mat_transformed[test_index, :].astype('float64')
    
    # print("train_index", train_index)
    # print("test_index", test_index)
    
    train_set = roc_mat_red[train_index, :].astype('float64')
    train_set[np.isnan(train_set)] = 0
    
    test_set = roc_mat_red[test_index, :].astype('float64')
    test_set[np.isnan(test_set)] = 0
    
    test_dataset_headers.extend(np.asarray(mat_file_list)[test_index])
    train_dataset_headers.extend(np.asarray(mat_file_list)[train_index])

    
    train_performance = np.sum(train_set, axis=0).reshape(1, n_configs)
    
    best_train_index = np.nanargmax(train_performance)

    
    ###################################
    header_mat = np.empty([len(test_index), 23], dtype="object")
    header_mat[:, 0] = np.full([1, len(test_index)], config_headers[iForest1])
    header_mat[:, 1] = np.full([1, len(test_index)], config_headers[iForest2])
    header_mat[:, 2] = np.full([1, len(test_index)], config_headers[LOF1])
    header_mat[:, 3] = np.full([1, len(test_index)], config_headers[best_train_index])
    
    roc_comp_mat = np.zeros([len(test_index), 23])
    roc_comp_mat[:, 0] = test_set[:, iForest1]
    roc_comp_mat[:, 1] = test_set[:, iForest2]
    roc_comp_mat[:, 2] = test_set[:, LOF1]
    roc_comp_mat[:, 3] = test_set[:, best_train_index]
    

    # add best per data
    for i in range(len(test_index)):


        test_data_index = data_index[test_index[i]]
        test_data_header = data_headers[test_index[i]]


        train_data_index = Diff(np.where(data_index==test_data_index)[0].tolist(), [test_index[i]])
        train_data_performance = roc_mat_red[train_data_index, :]
        train_data_performance_individual = np.nanargmax(train_data_performance, axis=1)
        train_data_best_index = np.nanargmax(np.sum(train_data_performance, axis=0).reshape(1, n_configs))
        
        roc_comp_mat[i, 4] = test_set[i, train_data_best_index]
        header_mat[i, 4] = config_headers[train_data_best_index]

        roc_comp_mat[i, 5] = np.nanmax(test_set[i,:])
        header_mat[i, 5] = config_headers[np.nanargmax(test_set[i,:])]
        
        roc_comp_mat[i, 6] = test_set[i, argmaxatn(test_set[i,:], 2)]
        roc_comp_mat[i, 7] = test_set[i, argmaxatn(test_set[i,:], 3)]
        roc_comp_mat[i, 8] = test_set[i, argmaxatn(test_set[i,:], 5)]
        roc_comp_mat[i, 9] = test_set[i, argmaxatn(test_set[i,:], 10)]
        
        header_mat[i, 6] = config_headers[argmaxatn(test_set[i,:], 2)]
        header_mat[i, 7] = config_headers[argmaxatn(test_set[i,:], 3)]        
        header_mat[i, 8] = config_headers[argmaxatn(test_set[i,:], 5)]
        header_mat[i, 9] = config_headers[argmaxatn(test_set[i,:], 10)]
    
    # ALORS MSE
    best_k = (40,4)
    EMF = ExplicitMF(train_set, test_set, n_factors=best_k[0], learning='sgd')
    EMF.train(n_iter=100, meta_features=train_meta_transformed, valid_meta=test_meta_transformed, learning_rate=0.05, max_depth=best_k[1])
    U = EMF.user_vecs
    V = EMF.item_vecs
    bias_global = EMF.global_bias
    bias_user = EMF.user_bias
    bias_item = EMF.item_bias
    
    # print(EMF.regr_multirf.predict(test_meta).shape)
    predicted_scores = EMF.predict_new(test_meta_transformed)
    predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 10] = test_set[i, predicted_scores_max[i]]
        header_mat[i, 10] = config_headers[predicted_scores_max[i]]
        
        # get the top 3 index
        temp_index = argmaxn(predicted_scores[i, ], n=3)
        roc_comp_mat[i, 18] = test_set[i, temp_index].mean()
        header_mat[i, 18] = config_headers[temp_index[0]] + '|' + config_headers[temp_index[1]] + '|' + config_headers[temp_index[2]]
        
    #################################### ISAC
    clustering = KMeans(n_clusters=5)
    clustering.fit(train_meta_transformed)
    train_clusters = clustering.labels_
    predicted_clusters = clustering.predict(test_meta_transformed)
    
    for i in range(len(test_index)):
        train_data_index = np.where(train_clusters==predicted_clusters[i])[0]
        train_data_performance = train_set[train_data_index, :]
        train_data_performance_individual = np.nanargmax(np.sum(train_data_performance, axis=0))
        
        roc_comp_mat[i, 11] = test_set[i, train_data_performance_individual]
        header_mat[i, 11] = config_headers[train_data_performance_individual]
    ###########################################################################
    # ALGOSMART
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(train_meta_transformed)
    
    neighbors = neigh.kneighbors(test_meta_transformed, 5, return_distance=False)
    # only select the k==1 case one
    k1_neighbors = neighbors[:, 0]
    
    train_data_performance = train_set[k1_neighbors, :]
    train_data_performance_individual = np.nanargmax(train_data_performance, axis=1)
    for i in range(len(test_index)):
        roc_comp_mat[i, 12] = test_set[i, train_data_performance_individual[i]]
        header_mat[i, 12] = config_headers[train_data_performance_individual[i]]
    

    ##########################################################################
    # MetaOD_C
    best_k = 40

    train_meta_transformed = meta_mat_transformed[train_index, :]
    test_meta_transformed =  meta_mat_transformed[test_index, :]
    
    train_roc = roc_mat_red[train_index, :].astype('float64')
    train_roc[np.isnan(train_roc)] = 0
    test_roc = roc_mat_red[test_index, :].astype('float64')
    test_roc[np.isnan(test_roc)] = 0
    
    # constuct the combined matrix
    combined_matrix = np.concatenate([train_roc, train_meta_transformed], axis=1)
    scalar = MinMaxScaler()
    combined_matrix_transformed =scalar.fit_transform(combined_matrix) 
    
    combined_matrix_test = np.concatenate([np.zeros([test_roc.shape[0], test_roc.shape[1]]), test_meta_transformed], axis=1)
    
    # how to do the transformation
    combined_matrix_test_transformed = scalar.transform(combined_matrix_test)
    combined_matrix_test_transformed_2 = MinMaxScaler().fit_transform(combined_matrix_test)
    combined_matrix_test_transformed_3 = combined_matrix_test_transformed
    combined_matrix_test_transformed_3[:, :n_configs] = 0
    
    
    u1, s1, vt1 = svd(combined_matrix_transformed)
    
    k = best_k 
    u1k = u1[:, :k]
    s1k = diag(s1[:k])
    vt1_k = vt1[:k, :]
    
    predicted_test_matrix = np.dot(np.dot(combined_matrix_test_transformed_3, vt1_k.T), vt1_k)
    predicted_test_matrix_max = np.nanargmax(predicted_test_matrix[:, :n_configs], axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 13] = test_set[i, predicted_test_matrix_max[i]]
        header_mat[i, 13] = config_headers[predicted_test_matrix_max[i]]
    
    print(ndcg_score(test_roc, predicted_test_matrix[:, :n_configs], k=n_configs))
    
    ########################################################
    # SS
    X_test_index = []
    for i in test_index:
        X_test_index.extend(list(range(i*n_configs, (i+1)*n_configs)))

    X_train_index = Diff(list(range(len(y_reg_fixed))), X_test_index)
    
    X_train = X_reg_fixed[X_train_index, :]
    y_train = y_reg_fixed[X_train_index]


    X_train, y_train = shuffle(X_train, y_train)
    
    clf = xgb.XGBRegressor(objective='reg:tweedie')
                           
    clf.fit(X_train, y_train)
    

    for i in range(len(test_index)):
        X_test_index = list(range(test_index[i]*n_configs, (test_index[i]+1)*n_configs))
        X_test = X_reg_fixed[X_test_index, :]
        # y_test = y_reg_fixed[X_test_index]    
        
        test_pred = clf.predict(X_test).reshape(1,-1)
        test_pred[test_pred < 0] = 0
        # print(test_pred)
        
        predicted_test_matrix_max = np.nanargmax(test_pred, axis=1)
        predicted_test_std = np.std(test_pred, axis=1)
        
        # no score
        if predicted_test_std <= 0.01:
            roc_comp_mat[i, 14] = test_set[i, iForest2]
            header_mat[i, 14] = config_headers[iForest2]
        else:    
            roc_comp_mat[i, 14] = test_set[i, predicted_test_matrix_max[0]]
            header_mat[i, 14] = config_headers[predicted_test_matrix_max[0]]
    
    ##########################################################################
    from MF_SGD_Fixed import ExplicitMFF
    # ALORS Fixed
    best_k = 40
    # CV ends
    # EMF = ExplicitMF(train_set, n_factors=1, learning='sgd')
    EMFF = ExplicitMFF(train_set, n_factors=best_k, learning='sgd')
    EMFF.train(n_iter=80, meta_features=train_meta_transformed, learning_rate=0.05)
    # EMF.train(n_iter=30, meta_features=train_meta_transformed, learning_rate=0.05, max_depth=3)
    U = EMFF.user_vecs
    V = EMFF.item_vecs
    bias_global = EMFF.global_bias
    # bias_user = EMF.user_bias
    bias_item = EMFF.item_bias
    
    # print(EMF.regr_multirf.predict(test_meta).shape)
    predicted_scores = EMFF.predict_new(test_meta_transformed)
    predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 15] = test_set[i, predicted_scores_max[i]]
        header_mat[i, 15] = config_headers[predicted_scores_max[i]]
        
    ##########ALORS Rank######################################################
    # ALORS Rank Sigmoid
    from MF_SGD_Rank_Sigmoid import ExplicitMFRank
    best_k = (40,4)
    # CV ends
    # EMF = ExplicitMF(train_set, n_factors=1, learning='sgd')
    EMF = ExplicitMFRank(train_set, valid_ratings=test_set, n_factors=best_k[0], learning='sgd')
    # EMF = ExplicitMFRank(train_set, test_set, n_factors=best_k[0], learning='sgd')
    EMF.train(n_iter=50, meta_features=train_meta_transformed, valid_meta=test_meta_transformed, learning_rate=0.1, max_depth=best_k[1], max_rate=0.95, min_rate=0.05, discount=1, n_steps=8)
    # EMF.train(n_iter=150, meta_features=train_meta_transformed, valid_meta=test_meta_transformed, learning_rate=0.8, max_depth=best_k[1])
    # EMF.train(n_iter=30, meta_features=train_meta_transformed, learning_rate=0.05, max_depth=3)
    U = EMF.user_vecs
    V = EMF.item_vecs
    
    # print(EMF.regr_multirf.predict(test_meta).shape)
    predicted_scores = EMF.predict_new(test_meta_transformed)
    predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 16] = test_set[i, predicted_scores_max[i]]
        header_mat[i, 16] = config_headers[predicted_scores_max[i]]

        # get the top 3 index
        temp_index = argmaxn(predicted_scores[i, ], n=3)
        roc_comp_mat[i, 19] = test_set[i, temp_index].mean()
        header_mat[i, 19] = config_headers[temp_index[0]] + '|' + config_headers[temp_index[1]] + '|' + config_headers[temp_index[2]]
    
    #############################################################################
    # MetaOD_F
    from MF_SGD_Rank_Fixed_Sigmoid import ExplicitMFRankF
    best_k = 40
    # CV ends
    # EMF = ExplicitMF(train_set, n_factors=1, learning='sgd')
    EMFF = ExplicitMFRankF(train_set, valid_ratings=test_set, n_factors=best_k, learning='sgd')
    EMFF.train(n_iter=50, meta_features=train_meta_transformed, valid_meta=test_meta_transformed, learning_rate=0.05, max_rate=1, min_rate=0.05, discount=1, n_steps=8)
    U = EMFF.user_vecs
    V = EMFF.item_vecs
    predicted_scores = EMFF.predict_new(test_meta_transformed)
    predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 17] = test_set[i, predicted_scores_max[i]]
        header_mat[i, 17] = config_headers[predicted_scores_max[i]]

        # get the top 3 index
        temp_index = argmaxn(predicted_scores[i, ], n=3)
        roc_comp_mat[i, 20] = test_set[i, temp_index].mean()
        header_mat[i, 20] = config_headers[temp_index[0]] + '|' + config_headers[temp_index[1]] + '|' + config_headers[temp_index[2]]
        
    #############################################################################
    # MetaOD
    from MetaOD import ExplicitMFRankF
    best_k = 40
    # CV ends

    EMFF = ExplicitMFRankF(train_set, valid_ratings=test_set, n_factors=best_k, learning='sgd')
    EMFF.train(n_iter=50, meta_features=train_meta_transformed, valid_meta=test_meta_transformed, learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1, n_steps=8)

    U = EMFF.user_vecs
    V = EMFF.item_vecs

    # print(EMF.regr_multirf.predict(test_meta).shape)
    predicted_scores = EMFF.predict_new(test_meta_transformed)
    predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
    
    for i in range(len(test_index)):
        roc_comp_mat[i, 21] = test_set[i, predicted_scores_max[i]]
        header_mat[i, 21] = config_headers[predicted_scores_max[i]]

        # get the top 3 index
        temp_index = argmaxn(predicted_scores[i, ], n=3)
        roc_comp_mat[i, 22] = test_set[i, temp_index].mean()
        header_mat[i, 22] = config_headers[temp_index[0]] + '|' + config_headers[temp_index[1]] + '|' + config_headers[temp_index[2]]
    ###########################################################################

    if ind == n_folds-1:
        roc_cv[ind*n_datasets_fold:, :] = roc_comp_mat
        header_cv[ind*n_datasets_fold:, :] = header_mat
    else:
        roc_cv[ind*n_datasets_fold: (ind+1)*n_datasets_fold, :] = roc_comp_mat
        header_cv[ind*n_datasets_fold: (ind+1)*n_datasets_fold, :] = header_mat    
    
    for j in all_perm:
        print(headers[j[0]], '|',  headers[j[1]], '|', 
              wilcoxon(roc_comp_mat[:, j[0]], roc_comp_mat[:, j[1]], zero_method='zsplit')[0], '|', 
              wilcoxon(roc_comp_mat[:, j[0]], roc_comp_mat[:, j[1]], zero_method='zsplit')[1], '|', 
              int(wilcoxon(roc_comp_mat[:, j[0]], roc_comp_mat[:, j[1]], zero_method='zsplit')[1]<=0.05))
    
result_header = np.c_[test_dataset_headers, header_cv]
result= np.c_[test_dataset_headers, roc_cv]
heads = np.asarray(['dataset', 'iForest (200, 0.1)', 'iForest (150, 0.5)', 
           '(LOF (20, minkowski)', 'GB', 
           'EUB', '1st Per Data', '2nd Per Data', '3rd Per Data', '5th Per Data', '10th Per Data', 
           'ALORS_MSE', 'ISAC', 'AS', 'MetaOD_C', 'SS', 'ALORS_F',  
           'ALORS_Rank', 'ALORS_Rank_F', 'ALORS_F_3', 'ALORS_Rank_3', 'ALORS_Rank_F_3',  'MetaOD', 'MetaOD_3']).reshape(1, 24)

result_header = np.r_[heads, result_header]
result = np.r_[heads, result]

print()
print()
print("full")
for j in all_perm:
    print(headers[j[0]], '|',  headers[j[1]], '|', wilcoxon(roc_cv[:, j[0]], roc_cv[:, j[1]])[0], 
          '|', wilcoxon(roc_cv[:, j[0]], roc_cv[:, j[1]])[1], 
          '|', int(wilcoxon(roc_cv[:, j[0]], roc_cv[:, j[1]])[1]<=0.05), '|', np.mean(roc_cv[:, j[0]]), '|', np.mean(roc_cv[:, j[1]]))
