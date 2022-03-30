import numpy as np
import pandas as pd


from sklearn.tree import DecisionTreeRegressor as DTR, DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb



from sklearn.datasets import make_regression, make_classification, make_circles, load_boston
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, KFold


from catboost import CatBoostRegressor
from catboost import Pool
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import catboost
from numpy.testing import assert_almost_equal
from xgboost import XGBRegressor