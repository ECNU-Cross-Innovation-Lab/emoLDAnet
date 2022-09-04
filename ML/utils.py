import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

lda_clip = {-3:3,-2:1,-1:4}
norm = True
def label_D(x):
    if x<0.1587:
        x = 0
    elif x>=0.1587 and x<0.2381:
        x = 1
    elif x>=0.2381 and x<0.3968:
        x = 2
    elif x>= 0.3968:
        x = 3
    if norm:
        x /= 3
    return x

def label_A(x):
    if x<0.2677:
        x = 0
    elif x>= 0.2677:
        x = 1
    return x

def label_L(x):
    if x<0.1333:
        x = 0
    elif x>=0.1333 and x<0.2167:
        x = 1
    elif x>=0.2167 and x<0.3167:
        x = 2
    elif x>=0.3167 and x<0.4:
        x = 3
    elif x>= 0.4:
        x = 4
    if norm:
        x /= 4
    return x


def to_int(pred, y_index):
    if norm:
        pred = np.multiply(lda_clip[y_index],pred)
    pred = np.clip(pred, 0, lda_clip[y_index])
    pred = np.around(pred)
    return pred

def get_model(name):
    if name == 'forest':
        model = RandomForestRegressor()
    elif name == 'svm':
        model = SVR()
    elif name == 'xgb':
        model = XGBRegressor(n_estimators=100,max_depth=1,reg_alpha=0.015,reg_lambda=0.01,learning_rate=0.05,min_child_weight=1)
    elif name == 'gbt':
        model = GradientBoostingRegressor(n_estimators=100,max_depth=10,learning_rate=0.15)
    elif name == 'knn':
        model = KNeighborsRegressor()
    elif name == 'tree':
        model = DecisionTreeRegressor()
    elif name == 'mlp':
        model = MLPRegressor()
    elif name == 'lasso':
        model = Lasso()
    elif name == 'ridge':
        model = RidgeCV()
    return model
