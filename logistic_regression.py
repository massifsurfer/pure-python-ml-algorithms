import numpy as np
import pandas as pd
import random


class MyLogReg:

    eps = 1e-15

    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        
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
        self.current_X, self.current_y = None, None
    
    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def __get_metric_score(self, X, y):
        pred_labels = self.predict(X)
        y_prob = self.predict_proba(X)


        TP = (pred_labels * y).sum()
        TN = ((pred_labels ^ 1) * (y ^ 1)).sum()
        FP = (pred_labels * (y ^ 1)).sum()
        FN = ((pred_labels ^ 1) * y).sum()

        if self.metric == "accuracy":
            return (TP + TN) / (TP + TN + FP + FN)
        elif self.metric == "precision":
            return TP / (TP + FP)
        elif self.metric == "recall":
            return TP / (TP + FN)
        elif self.metric == "f1":
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            return 2 * precision * recall / (precision + recall)
        elif self.metric == "roc_auc":
            y_prob_class = np.concatenate([np.around(y_prob.reshape(-1, 1), 10), y.reshape(-1, 1)], axis = 1)
            self.q = y_prob_class
            y_prob_class = y_prob_class[np.argsort(-y_prob_class[:, 0])]
            self.g = y_prob_class
            summ = 0
            for i in range(len(y_prob_class)):
                if y_prob_class[i][1] == 0:
                    higher_pos_score = 0
                    same_pos_score = 0
                    for j in range(i):
                        if y_prob_class[j][1] == 1:
                            if y_prob_class[j][0] > y_prob_class[i][0]:
                                higher_pos_score += 1
                            elif y_prob_class[j][0] == y_prob_class[i][0]:
                                same_pos_score += 0.5
                    summ += higher_pos_score + same_pos_score
                    
            roc_auc = summ / (y.sum() * (y ^ 1).sum())
            return roc_auc

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        X, y = X.to_numpy(), y.to_numpy()
        self.current_X, self.current_y = X, y
        X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X), axis=1)
        self.weigts = np.ones(X.shape[1])
        
        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self.__get_batch(X, y)
            batch_size = X_batch.shape[0]

            y_logit = np.dot(X_batch, self.weigts)


            pred_labels = 1 / (1 + np.exp(-y_logit))

            MSE_grad = 1 / batch_size * np.dot((pred_labels - y_batch) , X_batch)
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
            
            if verbose:
                if i % verbose == 0:
                    self.metric_score = self.__get_metric_score(X, y)
                    print(f"iteration {i}, metric ({self.metric}): {self.metric_score}, weigts: {self.weigts}")
    
    def get_coef(self):
        return self.weigts[1:]

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.weigts)))
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def get_best_score(self):
        X = np.concatenate((np.ones(self.current_X.shape[0]).reshape(-1, 1), self.current_X), axis=1)
        self.metric_score = self.__get_metric_score(X, self.current_y)
        return self.metric_score
    
    def __get_batch(self, X, y):
        if type(self.sgd_sample) == float:
            sample_index = random.sample(range(X.shape[0]), round(X.shape[0] * self.sgd_sample))
        elif type(self.sgd_sample) == int:
            sample_index = random.sample(range(X.shape[0]), self.sgd_sample)
        else:
            sample_index = np.arange(X.shape[0])
        return X[sample_index], y[sample_index]
        

    