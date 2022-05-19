from IPython.display import Image 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sns


from sklearn.datasets import make_regression, make_classification, make_circles, load_boston
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, KFold

