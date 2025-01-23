import numpy as np
import pandas as pd
import random


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weigts = None

        self.metric = metric
        self.metric_score = None

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.dynamic_lr = callable(learning_rate)

        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def __get_metric_score(self, X, y):
        pred_labels = np.dot(X, self.weigts)

        if self.metric == "mae":
            return np.mean(np.abs(y - pred_labels))
        elif self.metric == "mse":
            return np.mean((y - pred_labels) ** 2)
        elif self.metric == "rmse":
            return np.sqrt(np.mean((y - pred_labels) ** 2))
        elif self.metric == "mape":
            return 100 * np.mean(np.abs((y - pred_labels) / y))
        elif self.metric == "r2":
            return 1 - np.sum((y - pred_labels) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        X, y = X.to_numpy(), y.to_numpy()
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        self.weigts = np.ones(X.shape[1])
        
        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self.__get_batch(X, y)
            batch_size = X_batch.shape[0]

            pred_labels = np.dot(X_batch, self.weigts)

            MSE_grad = 2 / batch_size * np.dot((pred_labels - y_batch) , X_batch)
            if self.reg == "l1":
                MSE_grad = MSE_grad + self.l1_coef * np.sign(self.weigts)
            elif self.reg == "l2":
                MSE_grad = MSE_grad + 2 * self.l2_coef * self.weigts
            elif self.reg == "elasticnet":
                MSE_grad = MSE_grad + self.l1_coef * np.sign(self.weigts) + 2 * self.l2_coef * self.weigts

            if self.dynamic_lr:
                current_lr = self.learning_rate(i)
            else:
                current_lr = self.learning_rate

            self.weigts = self.weigts - current_lr * MSE_grad
            self.metric_score = self.__get_metric_score(X, y)
            
            if verbose:
                if i % verbose == 0:                    
                    print(f"iteration {i}, metric ({self.metric}): {self.metric_score}, weigts: {self.weigts}")
    
    def get_coef(self):
        return self.weigts[1:]

    def predict(self, X):
        X = X.to_numpy()
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        return np.dot(X, self.weigts)
    
    def get_best_score(self):
        return self.metric_score
    
    def __get_batch(self, X, y):
        if type(self.sgd_sample) == float:
            sample_index = random.sample(range(X.shape[0]), round(X.shape[0] * self.sgd_sample))
        elif type(self.sgd_sample) == int:
            sample_index = random.sample(range(X.shape[0]), self.sgd_sample)
        else:
            sample_index = np.arange(X.shape[0])
        return X[sample_index], y[sample_index]
        
