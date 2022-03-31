import numpy as np
import xgboost as xgb
import copy
import os
import matplotlib
import re
import matplotlib.pyplot as plt
from itertools import repeat

import pandas as pd

def expected_bias_squared(expected_predictions, labels):
    bias_squared = np.square(expected_predictions - labels)
    return np.average(bias_squared)


def expected_variance(predictions, expected_predictions):
    squared_expected_predictions = np.square(expected_predictions)
    expected_squared_predictions = np.average(np.square(predictions), axis=0)
    return np.average(expected_squared_predictions - squared_expected_predictions)

def expected_mse(predictions, labels):
    preds = np.asarray(predictions)
    num_instances = len(labels)
    expected_mse_per_instance = np.zeros(num_instances)
    for i in range(num_instances):
        diff = labels[i] - preds[:, i]
        expected_mse_per_instance[i] = np.average(np.square(diff))
    return np.average(expected_mse_per_instance)

def plot_experiment(title, x_label, plot_x, results):
    labels = ["irreducible_error", "variance", "bias^2"]
    df = pd.DataFrame()
    for label in labels:
        df[label] = [res[label] for res in results]
    df[x_label] = plot_x
    df = df.set_index(x_label)
    df.plot.area()
    plt.title(title)
    plt.ylabel("MSE")
    plt.xlim(np.min(plot_x), np.max(plot_x))
    title = re.sub(' -', '', title)
    snake_title = re.sub(' ', '_', title + ' ' + x_label).lower()
    plt.grid()
    
def get_biasvar(base_model, generator, n=1000, n_test=10000, label_variance=1000.0, num_models=30):
    X_test, y_test, test_noise = generator(n_test, label_variance)
    models = []
    for i in range(num_models):
        X, y, noise = generator(n, label_variance)
        model = copy.deepcopy(base_model)
        models.append(model.fit(X, y))

    preds = []
    for model in models:
        pred = model.predict(X_test)
        preds.append(pred.astype(np.double))

    expected_predictions = np.average(preds, axis=0)

    bias_squared = expected_bias_squared(expected_predictions, y_test - test_noise)
    variance = expected_variance(preds, expected_predictions)
    mse = expected_mse(preds, y_test)
    return {"bias^2": bias_squared, "mse": mse, "variance": variance,
            "irreducible_error": label_variance}