from IPython.display import Image 
import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import BaggingRegressor #Вот наш бэггинг регрессор
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
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

def get_grid(data, border=1., step=.01): #получаем все точки плоскости
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step),
                       np.arange(y_min, y_max, step))

def plot_model(X_train, y_train, clf, title=None, proba=False):
    xx, yy = get_grid(X_train) #получаем все точки плоскости
    plt.figure(figsize=(7, 7))
    # предсказываем значения для каждой точки плоскости
    
    if proba: # нужно ли предсказывать вероятности 
        predicted = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    else:
        predicted = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Отрисовка плоскости
    ax = plt.gca()
    ax.pcolormesh(xx, yy, predicted, cmap='spring')
    
    # Отрисовка точек
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, cmap='spring', edgecolors='b')
    colors = ['purple', 'yellow', 'orange']
    patches = []
    for yi in np.unique(y_train):
        patches.append(mpatches.Patch(color=colors[int(yi)], label='$y_{pred}=$'+str(int(yi))))
    ax.legend(handles=patches)
    plt.title(title)
    return clf