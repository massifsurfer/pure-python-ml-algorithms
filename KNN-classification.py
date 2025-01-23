from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import random


class MyKNNClf:
    def __init__(self, k=3, metric="euclidean", weight="uniform"):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def fit(self, X, y):
        self.X, self.y = X.to_numpy(), y.to_numpy().astype(int)
        self.train_size = self.X.shape

    def __calc_metric(self, x):
        if self.metric == 'manhattan':
            return cdist(self.X, np.expand_dims(x, axis=0), metric='cityblock').squeeze()
        return cdist(self.X, np.expand_dims(x, axis=0), metric=self.metric).squeeze()
    
    def __pred_label_weights(self, x):
        metrics = self.__calc_metric(x)
        k_nearest_neighbors = np.argsort(metrics)[:self.k]
        k_nearest_neighbors_labels = self.y[k_nearest_neighbors]
        k_nearest_neighbors_metrics = metrics[k_nearest_neighbors]
        

        if self.weight == "rank":
            normalization_coef = 1 / (1 / np.arange(1, self.k + 1)).sum()
            q_0 = normalization_coef * ((k_nearest_neighbors_labels ^ 1) / (np.arange(1, self.k + 1))).sum()
            q_1 = normalization_coef * ((k_nearest_neighbors_labels) / (np.arange(1, self.k + 1))).sum()
        elif self.weight == "distance":
            normalization_coef = 1 / (1 / k_nearest_neighbors_metrics).sum()
            q_0 = normalization_coef * ((k_nearest_neighbors_labels ^ 1) / k_nearest_neighbors_metrics).sum()
            q_1 = normalization_coef * ((k_nearest_neighbors_labels) / k_nearest_neighbors_metrics).sum()
        else:
            label_counts = np.bincount(k_nearest_neighbors_labels)
            if label_counts.shape[0] == 1:
                q_0 = 1
                q_1 = 0
            else:
                q_0 = label_counts[0] / self.k
                q_1 = label_counts[1] / self.k
        return np.array([q_0, q_1])
            


    def predict(self, X_to_predict):
        X_to_predict = X_to_predict.to_numpy()
        pred_labels = []
        for x in X_to_predict:
            labels_weights = self.__pred_label_weights(x)
            if labels_weights[0] == labels_weights[1]:
                pred_labels.append(1)
            else:
                pred_labels.append(np.argmax(self.__pred_label_weights(x)))
        return np.array(pred_labels)
    
    def predict_proba(self, X_to_predict):
        X_to_predict = X_to_predict.to_numpy()
        positive_probas = []
        for x in X_to_predict:
            positive_probas.append(self.__pred_label_weights(x)[1])
        return np.array(positive_probas)

    def __str__(self):
        return f"MyKNNClf class: k={self.k}"