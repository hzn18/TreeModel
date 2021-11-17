import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from gbdt import GradientBoostingRegressor

if __name__ == "__main__":
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_name = boston.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_df = pd.DataFrame(X_train, columns=feature_name)
    y_train_df = pd.Series(y_train)
    model = GradientBoostingRegressor(2,2)
    model.fit(X_train_df, y_train_df)
    X_test_df = pd.DataFrame(X_test, columns=feature_name)
    print(model.predict(X_test_df)[0:20])
    print(y_test[0:20])
