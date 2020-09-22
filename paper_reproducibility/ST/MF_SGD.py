
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from numpy.linalg import solve
from sklearn.linear_model import Lasso

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import dcg_score, ndcg_score

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

class ExplicitMF():
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
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose
        self.train_loss_ = [0]
        self.valid_loss_ = [0]

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI), 
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, meta_features, valid_meta=None, n_iter=10, learning_rate=0.1, n_estimators=100, max_depth=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))

        self.learning_rate = learning_rate
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])


        ctr = 1
        np_ctr = 1
        while ctr <= n_iter:
            
            self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, n_jobs=4))
            
            #make sure it is non zero            
            self.user_vecs[np.isnan(self.user_vecs)] = 0
            
            self.regr_multirf.fit(meta_features, self.user_vecs)
            
            meta_valid_scaled = self.regr_multirf.predict(valid_meta)
            
            ndcg_s = []
            for w in range(self.ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.ratings[w,:]], [np.dot(self.user_vecs[w,:], self.item_vecs.T)], k=self.n_items))

            
            # print('ALORS Rank iteration', ctr, ndcg_score(self.ratings, np.dot(self.user_vecs, self.item_vecs.T)))
            # print('ALORS Rank iteration', ctr, 'training', np.mean(ndcg_s))
            self.train_loss_.append(np.mean(ndcg_s))
            
    
            ndcg_s = []
            for w in range(self.valid_ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.valid_ratings[w,:]], [np.dot(meta_valid_scaled[w,:], self.item_vecs.T)], k=self.n_items))
            
            # print('ALORS iteration', ctr, ndcg_score(self.ratings, np.dot(self.user_vecs, self.item_vecs.T)))
            self.valid_loss_.append(np.mean(ndcg_s))
            print('ALORS MSE iteration', ctr, 'train', self.train_loss_[-1], 'valid', self.valid_loss_[-1])
            
            if ((self.valid_loss_[-1] - self.valid_loss_[-2])/self.valid_loss_[-2]) <= 0.001:
                # print(((self.valid_loss_[-1] - self.valid_loss_[-2])/self.valid_loss_[-2]))
                np_ctr += 1
            else:
                np_ctr = 1 
            if np_ctr > 3:
                break
            
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            
            for idx in self.training_indices:
                u = self.sample_row[idx]
                i = self.sample_col[idx]
                prediction = self.predict(u, i)
                e = (self.ratings[u,i] - prediction) # error
                
                # Update biases
                self.user_bias[u] += self.learning_rate * \
                                    (e - self.user_bias_reg * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * \
                                    (e - self.item_bias_reg * self.item_bias[i])
                
                #Update latent factors
                self.user_vecs[u, :] += self.learning_rate * \
                                        (e * self.item_vecs[i, :] - \
                                         self.user_fact_reg * self.user_vecs[u,:])
                self.item_vecs[i, :] += self.learning_rate * \
                                        (e * self.user_vecs[u, :] - \
                                         self.item_fact_reg * self.item_vecs[i,:])
            ctr += 1
        
        return self
    

    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter
    
    
        
    def predict_new(self, test_meta):
        test_meta = check_array(test_meta)
        # assert (test_meta.shape[1]==200)
        
        test_k = self.regr_multirf.predict(test_meta)
        assert (test_k.shape[1] == self.n_factors)
        
        predicted_scores = np.dot(test_k, self.item_vecs.T) + self.item_bias
        
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


# EMF = ExplicitMF(train_roc_cv, valid_roc_cv, n_factors=3, learning='sgd', verbose=False)
# EMF.train(n_iter=500, meta_features=train_data_cv, valid_meta=valid_data_cv, learning_rate=0.1)

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