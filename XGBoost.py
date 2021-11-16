"""
@Author: houzhinan
@Date: 2021-11-15 22:21
@Last Modified by: houzhinan
@Last Modified time: 2021-11-16 18:58
"""

import numpy as np
from numpy as ndarray
from queue import Queue
from RegressionTree import TreeNode

class RegressionTree:
    """ build a XGBoost Tree

    Attributes:
        root (TreeNode): root node of regression tree

    """
    def __init__(self, max_depth):
        self.root = TreeNode(0)
        self.max_depth = max_depth
                

    def fit(self, x_data, y_data):
        features = x_data.columns
        now_depth = 0
        leaf_queue = Queue()
        leaf_queue.put((root, x_data.index))
        while(now_depth < max_depth):
            region_number = leaf_queue.qsize()
            for i in range(region_number):
                father_node, region = leaf_queue.get()
                if len(region) == 0:
                    continue
                split_feature, split_value, left_value, right_value, left_region, right_region = self.split_region(x_data.loc[region], y_data.loc[region], features)
                left_node, right_node = father_node.split(split_feature, split_value, left_value, right_value)
                # 新的叶子节点
                leaf_queue.put((left_node, left_region))
                leaf_queue.put((right_node, right_region))
            now_depth += 1

    def predict(x_data):
        y_pred = []
        for i in x_data.index:
            node = self.tree
            while node.left_node != None and node.right_node != None:
                if x_data.loc[i, node.split_feature] < node.split_value:
                    node = node.left_node
                else:
                    node = node.right_node
            y_pred.append(node.leaf_value)
        return y_pred

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



