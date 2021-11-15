from RegressionTree import RegressionTree
import numpy as np
import pandas as pd

class GradientBoostingTree():
    def __init__(self, n_estimators = 10, max_depth = 3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.init_val = None 

    def _get_init_val(self, y_data):
        return y_data.mean()

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
            prediction += pd.Series(self.trees[-1].predict(x_data))
            residual = self._get_residual(y_data, prediction)
    
    def predict(self, x_data):
        tree_prediction = np.sum([tree.predict(x_data) for tree in self.trees], axis = 0)
        return self.init_val + pd.Series(tree_prediction)
