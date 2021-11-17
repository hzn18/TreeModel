"""
@Author: houzhinan
@Date: -
@Last Modified by: houzhinan
@Last Modified time: 2021-11-16 18:56
"""

from queue import Queue


class TreeNode:
    """ 回归树的分裂节点

    Attributes:
        left_value: 叶子节点数值
        split_feature: 切分变量名
        split_value: 切分值
        left_node: 切分后的左子树
        right_node: 切分后的右子树
    """
    def __init__(self, leaf_value):
        self.leaf_value = leaf_value
        self.split_feature = None 
        self.split_value = None 
        self.left_node = None 
        self.right_node = None 
    
    def split(self, split_feature, split_value, left_value, right_value):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_node = TreeNode(left_value)
        self.right_node = TreeNode(right_value)
        return self.left_node, self.right_node

class RegressionTree:
    """ 回归树模型
    Attributes:
        tree: 回归树的头节点
        max_depth: 回归树的最大深度
    """
    def __init__(self, max_depth = 3):
        self.tree = None
        self.max_depth = max_depth

    def split_region(self, x_data, y_data, features):
        split_feature = None
        split_value = None
        left_value = None
        right_value = None
        left_region = None
        right_region = None
        cost_function = -1
        
        for feature in features:
            values = list(x_data[feature].sort_values())
            values.append(values[-1] + 1)
            for value in values:
                cost_function_temp = 0
                if len(y_data[x_data[feature] < value]) != 0:
                    left_value_temp = y_data[x_data[feature] < value].mean()
                    cost_function_temp += y_data[x_data[feature] < value].apply(lambda x: (x - left_value_temp)**2).sum()
                else:
                    left_value_temp = value
                if len(y_data[x_data[feature] >= value]) != 0:
                    right_value_temp = y_data[x_data[feature] >= value].mean()
                    cost_function_temp += y_data[x_data[feature] >= value].apply(lambda x: (x - right_value_temp)**2).sum()
                else:
                    left_value_temp = value
                if cost_function == -1 or cost_function_temp < cost_function:        
                    split_feature = feature
                    split_value = value
                    left_value = left_value_temp
                    right_value = right_value_temp 
                    left_region = y_data[x_data[feature] < value].index
                    right_region = y_data[x_data[feature] >= value].index
                    cost_function = cost_function_temp
        return split_feature, split_value, left_value, right_value, left_region, right_region

        
    def fit(self, x_data, y_data):
        """ 用回归树模型拟合训练集数据
        
        采用广度搜索

        Arguments:
            x_data:训练集的特征数据
            y_data:训练集的标签数据
        """
        # 初始化
        features = x_data.columns
        leaf_queue = Queue()
        now_depth = 0
        self.tree = TreeNode(0)
        leaf_queue.put((self.tree, x_data.index))   # 叶子节点和区域
        while(now_depth < self.max_depth):
            region_number = leaf_queue.qsize()    
            for i in range(region_number):     #对每个区域进行遍历
                father_node, region = leaf_queue.get()
                # 计算分裂结果
                if len(region) == 0:
                   continue
                split_feature, split_value, left_value, right_value, left_region, right_region = self.split_region(x_data.loc[region], y_data.loc[region], features)
                # 根据分裂结果，更新树模型
                left_node, right_node = father_node.split(split_feature, split_value, 
                                                          left_value, right_value)
                # 新的叶子节点
                leaf_queue.put((left_node, left_region))
                leaf_queue.put((right_node, right_region))
            now_depth += 1

    def predict(self, x_data):
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

