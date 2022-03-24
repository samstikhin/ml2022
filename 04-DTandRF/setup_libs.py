from IPython.display import Image 
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import BaggingRegressor #Вот наш бэггинг регрессор
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
# отключим всякие предупреждения Anaconda
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
