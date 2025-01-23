from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import random
from functools import reduce



class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = 'cityblock' if metric == 'manhattan' else metric

    def find_closest_clusters(self, X):
        X_dists = np.tril(cdist(X, X, metric=self.metric), -1)
        X_dists[X_dists == 0] = np.inf
        return np.unravel_index(X_dists.argmin(), X_dists.shape)

    
    def fit_predict(self, X):
        cluster_centroids = X.values
        cluster_points = [[X.index[i]] for i in range(X.shape[0])]
        while len(cluster_centroids) > self.n_clusters:
            closest_clusters = self.find_closest_clusters(cluster_centroids)

            cluster_points[closest_clusters[0]] += cluster_points[closest_clusters[1]]
            cluster_centroids[closest_clusters[0]] = X.loc[cluster_points[closest_clusters[0]]].to_numpy().mean(axis=0)
            
            cluster_points.pop(closest_clusters[1])
            cluster_centroids = np.delete(cluster_centroids, closest_clusters[1], axis=0)
        
        labeled_indexes = sorted(reduce(lambda x, y: x + y, [
            [(cluster_point, cluster_index) for cluster_point in cluster_points[cluster_index]]
            for cluster_index in range(len(cluster_points))
        ]), key=lambda x: x[0])
        labels = [labeled_index[1] for labeled_index in labeled_indexes]
        
        return labels

    def __str__(self):
        return f"MyAgglomerative class: n_clusters={self.n_clusters}"