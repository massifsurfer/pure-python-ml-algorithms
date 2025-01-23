from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd


class MyDBSCAN:
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = 'cityblock' if metric == 'manhattan' else metric

    def calc_dist(self, X):
        X_dists = cdist(X, X, metric=self.metric)
        X_dists[X_dists == 0] = np.inf
        return X_dists

    def check_point(
            self,
            X_dists,
            point_index,
            cluster_index,
            clusters,
            clusterized,
            possible_outliers
        ):
        if point_index in clusterized:
            return
        if point_index in possible_outliers:
            possible_outliers.remove(point_index)
        clusters.append((point_index, cluster_index))
        clusterized.append(point_index)
        neighbours = np.where(X_dists[point_index] < self.eps)[0]
        
        if neighbours.shape[0] >= self.min_samples:
            for neighbour_point_index in neighbours:
                self.check_point(
                    X_dists,
                    neighbour_point_index,
                    cluster_index,
                    clusters,
                    clusterized,
                    possible_outliers,
                )
        return


    def fit_predict(self, X):
        X_dists = self.calc_dist(X)
        possible_outliers = []
        clusterized = []
        clusters = []
        cluster_index = 0
        for i in range(X_dists.shape[0]):
            if np.count_nonzero(X_dists[i] < self.eps) >= self.min_samples:
                if i not in clusterized:
                    self.check_point(
                        X_dists,
                        i,
                        cluster_index,
                        clusters,
                        clusterized,
                        possible_outliers,
                    )
                    cluster_index += 1
                    #print(cluster_index)
            else:
                if i not in clusterized:
                    possible_outliers.append(i)
        #print(len(clusters))
        #print(list(zip(possible_outliers, [cluster_index] * len(possible_outliers))))
        clusters += list(zip(possible_outliers, [cluster_index] * len(possible_outliers)))
        return [point[1] for point in sorted(clusters, key=lambda x: x[0])]


    def __str__(self):
        return f"MyDBSCAN class: eps={self.eps}, min_samples={self.min_samples}"
        