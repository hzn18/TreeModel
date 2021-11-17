import numpy as np
import pandas as pd

from RegressionTree import RegressionTree


class GradientBoostingBase:
    def __init__(self, n_estimators = 10, max_depth = 3, learning_rate = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.init_val = None 

    def _get_init_val(self, y_data):
        raise NotImplementError

    def _get_residual(self, label, prediction):
        return label - prediction
    
    def fit(self, x_data, y_data):
        self.init_val = self._get_init_val(y_data)

        prediction = self.init_val

        residual = self._get_residual(y_data, self.init_val)

        for _ in range(self.n_estimators):
            tree = RegressionTree(self.max_depth)
            tree.fit(x_data, residual)
            self.trees.append(tree)

            #更新残差
            prediction = self.predict(x_data)
            residual = self._get_residual(y_data, prediction)
    
    def prediction(x_data):
        raise NotImplementError


class GradientBoostingRegressor(GradientBoostingBase):

    def _get_init_val(self, y_data):
        return y_data.mean()

    def predict(self, x_data):
        tree_prediction = np.sum([tree.predict(x_data) for tree in self.trees], axis = 0) 
        return self.init_val + self.learning_rate * pd.Series(tree_prediction)

class GradientBoostingClassifier(GradientBoostingBase):

    def _get_init_val(self, y_data):
        mean = y_data.mean()
        return np.log(mean / (1 - mean))

    def predict(self, x_data):
        tree_prediction = np.sum([tree.predict(x_data) for tree in self.trees], axis = 0)
        return (self.init_val + self.learning_rate * pd.Series(tree_prediction)).apply(lambda x: 1/(1 + np.exp(-x)))

