from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import random


class MyKNNReg:
    def __init__(self, k=3, metric="euclidean", weight="uniform" ):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __calc_metric(self, x):
        if self.metric == 'manhattan':
            return cdist(self.X, np.expand_dims(x, axis=0), metric='cityblock').squeeze()
        return cdist(self.X, np.expand_dims(x, axis=0), metric=self.metric).squeeze()
    

    def __calc_target(self, x):
        metrics = self.__calc_metric(x)
        k_nearest_neighbors = np.argsort(metrics)[:self.k]
        k_nearest_neighbors_y = self.y[k_nearest_neighbors]
        if self.weight == "uniform":
            target = k_nearest_neighbors_y.mean()
        elif self.weight == "rank":
            target = \
                (k_nearest_neighbors_y * ((1 / np.arange(1, self.k + 1)) / (1 / np.arange(1, self.k + 1)).sum())).sum()
        else:
            target = \
                (k_nearest_neighbors_y * ((1 / metrics[k_nearest_neighbors]) / (1 / metrics[k_nearest_neighbors]).sum())).sum()
        return target

    def fit(self, X, y):
        self.X, self.y = X.to_numpy(), y.to_numpy()
        self.train_size = self.X.shape[0]

    def predict(self, X_to_predict):
        X_to_predict = X_to_predict.to_numpy()
        pred_y = []
        for x_to_predict in X_to_predict:
            pred_y.append(self.__calc_target(x_to_predict))
        return np.array(pred_y)

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"