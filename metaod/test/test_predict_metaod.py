"""MetaOD prediction with the trained model
"""
# License: BSD 2 clause
import os
import unittest

from pyod.utils.data import generate_data
from metaod.models.utility import prepare_trained_model
from metaod.models.predict_metaod import select_model


class TestPredictMetaOD(unittest.TestCase):
    def setUp(self):
        self.contamination = 0.05  # percentage of outliers
        self.n_train = 1000  # number of training points
        self.n_test = 100  # number of testing points

        # Generate sample data
        self.X_train, self.y_train, self.X_test, self.y_test = \
            generate_data(n_train=self.n_train,
                          n_test=self.n_test,
                          n_features=3,
                          contamination=self.contamination,
                          random_state=42)

    def test_prepare_trained_model(self):
        # load pretrained models
        prepare_trained_model()
        print(os.path.join(os.getcwd(), "trained_models"))
        assert (os.path.isfile("trained_models.zip"))
        assert (os.path.isdir("trained_models"))

    def test_model_selection(self):
        prepare_trained_model()
        # recommended models
        selected_models = select_model(self.X_train, n_selection=100)
        assert ((len(selected_models) == 100))
