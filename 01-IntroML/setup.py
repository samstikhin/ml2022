import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

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

def plot_reg(X, y, clf_dtc, X_test):
    clf_dtc.fit(X, y)
    Y_test = clf_dtc.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, cmap='bwr', s=50, alpha=1)
    plt.plot(X_test, Y_test, color='r', alpha=1)
    plt.grid()