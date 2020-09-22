#%%
import numpy as np

from sklearn.metrics import dcg_score, ndcg_score 
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
from numpy.linalg import solve
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from numba import njit

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


@njit
def sgd(uh, vj, vi):
    temp_vt = np.exp(np.matmul(uh, (vj-vi)))
    return temp_vt

class ExplicitMFRank():
    def __init__(self, 
                 ratings,
                 valid_ratings,
                 n_factors=40,
                 learning='sgd',
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        learning : (str)
            Method of optimization. Options include 
            'sgd' or 'als'.
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.valid_ratings = valid_ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            # self.sample_row, self.sample_col = self.ratings.nonzero()
            # self.n_samples = len(self.sample_row)
            self.n_samples, self.n_configs = self.ratings.shape[0], self.ratings.shape[1]
        self._v = verbose
        self.train_loss_ = []
        self.valid_loss_ = []
    
    def train(self, meta_features, valid_meta=None, n_iter=10, learning_rate=0.1, n_estimators=100, max_depth=4, rho=1.1, alpha=0.5):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        

        self.learning_rate = learning_rate
        self.learning_rate_ = learning_rate
        
        last_train_ndcg = 0 
        
        ctr = 1
        while ctr <= n_iter:
            # if ctr % 10 == 0 and self._v:
            # print ('\tcurrent iteration: {}'.format(ctr))

            # self.learning_rate_ = self.learning_rate/np.sqrt(ctr)
            # self.learning_rate_ = self.learning_rate/np.cbrt(ctr)
            # self.learning_rate_ = self.learning_rate/np.cbrt(ctr)
            self.learning_rate_ = self.learning_rate/((1+0.02*ctr))
            
            self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=4))
            
            #make sure it is non zero            
            self.user_vecs[np.isnan(self.user_vecs)] = 0
            
            self.regr_multirf.fit(meta_features, self.user_vecs)
            
            meta_valid_scaled = self.regr_multirf.predict(valid_meta)
            
            ndcg_s = []
            for w in range(self.ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.ratings[w,:]], [np.dot(self.user_vecs[w,:], self.item_vecs.T)]))
            
            self.train_loss_.append(np.mean(ndcg_s))
            
            ###########################################################
            # adjust learning rate
            # if self.train_loss_[-1] > last_train_ndcg:
            #     self.learning_rate_ = self.learning_rate_*rho
            # else:
            #     self.learning_rate_ = self.learning_rate_*alpha
            
            # last_train_ndcg = self.train_loss_[-1]
            ###########################################################
    
            ndcg_s = []
            for w in range(self.valid_ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.valid_ratings[w,:]], [np.dot(meta_valid_scaled[w,:], self.item_vecs.T)]))
            
            # print('ALORS iteration', ctr, ndcg_score(self.ratings, np.dot(self.user_vecs, self.item_vecs.T)))
            self.valid_loss_.append(np.mean(ndcg_s))
            print('ALORS Rank iteration', ctr, 'train', self.train_loss_[-1], 'valid', self.valid_loss_[-1], 'learning rate', self.learning_rate_)
                
            train_indices = list(range(self.n_samples))
            np.random.shuffle(train_indices)
            # print(train_indices)
            
            for h in train_indices: 
                uh = self.user_vecs[h, :].reshape(1, -1)
                # print(uh.shape)
                grads = []
                
                for i in range(self.n_configs):
                # outler loop     
                    vi = self.item_vecs[i,:].reshape(-1, 1)
                    phis = []
                    rights = []
                    rights_v = []
                    # remove i from js 
                    js = list(range(self.n_configs))
                    js.remove(i)
                    
                    for j in js:
                        vj = self.item_vecs[j,:].reshape(-1, 1)
                        temp_vt = np.exp(np.matmul(uh, (vj-vi)))
                        temp_vt = np.ndarray.item(temp_vt)
                        # print(uh.re, (self.item_vecs[j,:]-self.item_vecs[i,:]).T.shape)
                        # print((self.item_vecs[j,:]-self.item_vecs[i,:]).reshape(-1, 1).shape)
                        # print(temp_vt.shape)
                        # assert (len(temp_vt)==1)
                        phis.append(temp_vt)
                        rights.append(temp_vt*(vj-vi))
                        rights_v.append(temp_vt*uh)
                    phi = np.sum(phis)+2
                    rights = np.asarray(rights).reshape(self.n_configs-1, self.n_factors)
                    rights_v = np.asarray(rights_v).reshape(self.n_configs-1, self.n_factors)
                    
                    # print(rights.shape, rights_v.shape)

                    right = np.sum(np.asarray(rights), axis=0)
                    right_v = np.sum(np.asarray(rights_v), axis=0)
                    # print(right, right_v)

                    # print(np.asarray(rights).shape, np.asarray(right).shape)
                    # grad = self.ratings[h, i] / (phi * (np.log(phi))**2)*right
                    # grad_v = self.ratings[h, i] / (phi * (np.log(phi))**2)*right_v
                    grad = (3**(self.ratings[h, i])-1) / (phi * (np.log(phi))**2)*right
                    grad_v = (3**(self.ratings[h, i])-1) / (phi * (np.log(phi))**2)*right_v
                    
                    self.item_vecs[i, :] +=  self.learning_rate_ * grad_v
                    
                    # print(h, i, grad.shape)
                    grads.append(grad)
                
                grads_uh = np.asarray(grads)
                grad_uh = np.sum(grads_uh, axis=0)
                
                self.user_vecs[h, :] -=  self.learning_rate_ * grad_uh
            
            ctr += 1
        
        return self

    def predict(self, u, i):
        """ Single user and item prediction."""
        # prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)        
        # prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction
    
    # def predict_all(self):
    #     """ Predict ratings for every user and item."""
    #     predictions = np.zeros((self.user_vecs.shape[0], 
    #                             self.item_vecs.shape[0]))
    #     for u in range(self.user_vecs.shape[0]):
    #         for i in range(self.item_vecs.shape[0]):
    #             predictions[u, i] = self.predict(u, i)
                
    #     return predictions
        
    def predict_new(self, test_meta):
        test_meta = check_array(test_meta)
        # assert (test_meta.shape[1]==200)
        
        test_k = self.regr_multirf.predict(test_meta)
        assert (test_k.shape[1] == self.n_factors)
        
        # predicted_scores = np.dot(test_k, self.item_vecs.T) + self.item_bias
        predicted_scores = np.dot(test_k, self.item_vecs.T)        
        #print(predicted_scores.shape)
        assert (predicted_scores.shape[0]== test_meta.shape[0])
        assert (predicted_scores.shape[1]==self.ratings.shape[1])
        
        return predicted_scores
    
#######################################
# random_state = np.random.RandomState(42)

# r = list(range(100))
# X = random_state.choice(r, size=[100, 5], replace=True)/100
# X_meta = random_state.choice(r, size=[100, 8], replace=True)

# X_train, X_test, X_train_meta, X_test_meta = train_test_split(X, X_meta, test_size=0.33, random_state=42)

# train_data_cv, valid_data_cv, train_roc_cv, valid_roc_cv = train_test_split(X_train_meta, X_train, test_size=0.2)


# EMF = ExplicitMFRank(train_roc_cv, valid_roc_cv, n_factors=3, learning='sgd', verbose=False)
# EMF.train(n_iter=500, meta_features=train_data_cv, valid_meta=valid_data_cv, learning_rate=0.5)

# U = EMF.user_vecs
# V = EMF.item_vecs

# pred_scores = np.dot(U, V.T)

# print('rating matrix size:', train_roc_cv.shape)
# print('Our modified loss and gradient results in NDCG:', ndcg_score(train_roc_cv, pred_scores))
# print()

# for j in range(10):
#     U = np.random.normal(size=U.shape)
#     V = np.random.normal(size=V.shape)
#     pred_scores = np.dot(U, V.T)

#     print('trial', j, 'random U, V result in NDCG:', ndcg_score(train_roc_cv, pred_scores))

# # bias_global = EMF.global_bias
# # bias_user = EMF.user_bias
# # bias_item = EMF.item_bias

# # # print(EMF.regr_multirf.predict(test_meta).shape)
# predicted_scores = EMF.predict_new(X_test_meta)
# # predicted_scores_max = np.nanargmax(predicted_scores, axis=1)