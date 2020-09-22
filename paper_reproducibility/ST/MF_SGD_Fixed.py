
import numpy as np
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
from numpy.linalg import solve
from sklearn.linear_model import Lasso

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import dcg_score, ndcg_score


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

class ExplicitMFF():
    def __init__(self, 
                 ratings,
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
        self.train_loss_ = []

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

    def train(self, meta_features, n_iter=10, learning_rate=0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vector
        # fixed the user vector by PCA

        # raise ValueError("A")
        self.pca = PCA(n_components=self.n_factors)
        self.pca.fit(meta_features)
        
        meta_features_pca = self.pca.transform(meta_features)
        
        self.scalar = MinMaxScaler()
        self.scalar.fit(meta_features_pca)
        
        meta_features_scaled = self.scalar.transform(meta_features_pca)

        self.user_vecs = meta_features_scaled
        
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        
        
        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            # self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)
        
        # self.regr_multirf = MultiOutputRegressor(RandomForestRegressor(
        #     n_estimators=n_estimators, max_depth=max_depth, n_jobs=4))

        # # self.regr_multirf = MultiOutputRegressor(Lasso()))
        # # self.regr_multirf = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators))
        
        # self.regr_multirf.fit(meta_features, self.user_vecs)
        return self
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            
            ndcg_s = []
            for w in range(self.ratings.shape[0]):
                ndcg_s.append(ndcg_score([self.ratings[w,:]], [np.dot(self.user_vecs[w,:], self.item_vecs.T)]))
            
            # print('ALORS Fixed iteration', ctr, ndcg_score(self.ratings, np.dot(self.user_vecs, self.item_vecs.T)))
            print('ALORS Fixed iteration', ctr, 'training', np.mean(ndcg_s))
            self.train_loss_.append(np.mean(ndcg_s))

            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                # self.user_vecs = self.als_step(self.user_vecs, 
                #                                self.item_vecs, 
                #                                self.ratings, 
                #                                self.user_fact_reg, 
                #                                type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            
            # Update biases
            # self.user_bias[u] += self.learning_rate * \
            #                     (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (e - self.item_bias_reg * self.item_bias[i])
            
            #Update latent factors
            # self.user_vecs[u, :] += self.learning_rate * \
            #                         (e * self.item_vecs[i, :] - \
            #                          self.user_fact_reg * self.user_vecs[u,:])
            self.item_vecs[i, :] += self.learning_rate * \
                                    (e * self.user_vecs[u, :] - \
                                     self.item_fact_reg * self.item_vecs[i,:])
    def predict(self, u, i):
        """ Single user and item prediction."""
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            prediction = self.global_bias + self.item_bias[i]
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
        # print('A', test_meta.shape)
    
        test_meta_scaled = self.pca.transform(test_meta)
        # print('B', test_meta_scaled.shape)
        
        test_meta_scaled = self.scalar.transform(test_meta_scaled)
        # print('c', test_meta_scaled.shape)
        # test_k = 
        # test_k = self.regr_multirf.predict(test_meta_scaled)
        # assert (test_k.shape[1] == self.n_factors)
        
        predicted_scores = np.dot(test_meta_scaled, self.item_vecs.T) + self.item_bias
        
        #print(predicted_scores.shape)
        assert (predicted_scores.shape[0]== test_meta.shape[0])
        assert (predicted_scores.shape[1]==self.ratings.shape[1])
        
        return predicted_scores

########################################
# random_state = np.random.RandomState(42)

# r = list(range(100))
# X = random_state.choice(r, size=[10, 5], replace=True)/100
# X_meta = random_state.choice(r, size=[10, 8], replace=True)

# X_train, X_test, X_train_meta, X_test_meta = train_test_split(X, X_meta, test_size=0.33, random_state=42)


# EMF = ExplicitMFF(X_train, n_factors=3, learning='sgd', verbose=False)
# EMF.train(n_iter=500, meta_features=X_train_meta, learning_rate=0.05)

# U = EMF.user_vecs
# V = EMF.item_vecs

# pred_scores = np.dot(U, V.T)

# print('rating matrix size:', X_train.shape)
# print('Our modified loss and gradient results in NDCG:', ndcg_score(X_train, pred_scores))
# print()

# for j in range(10):
#     U = np.random.normal(size=U.shape)
#     V = np.random.normal(size=V.shape)
#     pred_scores = np.dot(U, V.T)

#     print('trial', j, 'random U, V result in NDCG:', ndcg_score(X_train, pred_scores))

# # bias_global = EMF.global_bias
# # bias_user = EMF.user_bias
# # bias_item = EMF.item_bias

# # # print(EMF.regr_multirf.predict(test_meta).shape)
# predicted_scores = EMF.predict_new(X_test_meta)
# # predicted_scores_max = np.nanargmax(predicted_scores, axis=1)