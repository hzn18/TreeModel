"""
@Author: houzhinan
@Date: 2021-11-15 22:21
@Last Modified by: houzhinan
@Last Modified time: 2021-11-15 22:30
"""

import numpy as np
from numpy as ndarray


class TreeNode:
    """ build tree node

    Attributes:
        leaf_value (float): prediction of label
        split_feature (int): split by feature. columns index
        split_value (float): split point
        left_node (TreeNode): left child node
        right_node (TreeNode): right child node
    """
    def __init__(self, leaf_value, split_feature = None, split_value = None, left_node = None, right_node = None):
        self.leaf_value = leaf_value
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
    
    def __str__(self):
        #TODO:
        return 
    
    



class RegressionTree:
    """ build a XGBoost model

    Attributes:
        root (TreeNode): root node of regression tree

    """
    def __init__(self, max_depth):
        


class XGBoostModel:
    def __init__(self, max_depth, gamma, nabda):
        """[

        Args:
            max_depth ([type]): [description]
            gamma ([type]): [description]
            nabda ([type]): [description]
        """        
        self.max_depth = max_depth
        self.gamma = gamma
        self.nabda = nabda 

    def fit(x_data, y_data):
        #TODO:
        return 0

    def predict(x_data):
        #TODO:
        return 0



