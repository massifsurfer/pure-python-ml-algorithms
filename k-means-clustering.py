from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import random


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None

    def initialise_centroids(self, X):
        return np.array([
            [
                np.random.uniform(X[column].min(), X[column].max()) for column in X.columns
            ] for _ in range(self.n_clusters)
        ])
    
    def refine_centroids(self, centroids, clusters):
        return np.array(
        [clusters[i].mean(axis=0) if clusters[i].shape[0] != 0 else centroids[i] for i in range(len(clusters))]
        )
    
    def select_centroids(self, X):
        prev_centroids = self.initialise_centroids(X)
        for _ in range(self.max_iter):
            clusters = self.clusterise_data(X, prev_centroids)
            centroids = self.refine_centroids(prev_centroids, clusters)
            if np.allclose(prev_centroids, centroids):
                break
            prev_centroids = centroids
        return centroids

    
    def clusterise_data(self, X, centroids):
        clusters = [[] for _ in range(self.n_clusters)]
        for point in X.to_numpy():
            clusters[cdist(np.expand_dims(point, axis=0), centroids).argmin()].append(point)
        clusters = [np.array(cluster) for cluster in clusters]
        return clusters
    
    def calc_WCSS(self, X, centroids):
        WCSS = 0
        clusters = self.clusterise_data(X, centroids)
        for i in range(len(clusters)):
            if clusters[i].shape[0] == 0:
                continue
            WCSS += np.sum((clusters[i] - centroids[i])**2)
        return WCSS

    def fit(self, X):
        np.random.seed(seed=self.random_state)
        centroids_WCSS = []
        for i in range(1, self.n_init):
            centroids = self.select_centroids(X)
            WCSS = self.calc_WCSS(X, centroids)
            centroids_WCSS.append((centroids, WCSS))
        best_centroids, best_WCSS = min(centroids_WCSS, key=lambda x: x[1])
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_WCSS
        

    def predict(self, X_to_predict):
        X_to_predict = X_to_predict.to_numpy()
        predicted_clusters = []
        for x in X_to_predict:
            predicted_clusters.append(cdist(np.expand_dims(x, axis=0), self.cluster_centers_).argmin())
        return predicted_clusters

    def __str__(self):
        return f"MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"