#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:02:54 2019

@author: javenxu
"""

import linear_data_loader as ldl
import numpy as np
import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model

FEATURE_DIM = ldl.NUM_COLS
NUM_ACTIONS = 3
Q = 1



model = linear_model.Lasso(alpha=0.1)
model.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
model.coef_
clf.get_params()
q = 1
for i in range(1, K+1):
    for n in range(int(np.log(N)/np.log(2))):
        tmp = (np.power(2, n)-1) * K * q
        for j in range(q*(i-1)+1, q*i+1):
            T[i-1].append(tmp + j)


class Lasso_Bandit():
    def __init__(self, alpha, K=NUM_ACTIONS, q=Q, d=FEATURE_DIM):
        self.alpha = alpha
        self.K = K
        self.d = d
        self.T = [[] for t in range(self.K)]
        self.forced_sample_model = [linear_model.Lasso(alpha=alpha) for t in range(self.K)]
        self.all_sample_model = [linear_model.Lasso(alpha=alpha) for t in range(self.K)]
        self.q = q
        self.regret = []
        self.error_rate = []
        self.cumu_regret = 0
        self.sample_counter = 0
        self.samples = [[] for t in range(self.K)]

    def __str__(self):
        return "Lasso Bandit Algorithm"
    
    def train(self, data, labels):
        
        # initialize T
        N = data.shape[0]
        for i in range(1, self.K+1):
            for n in range(int(np.log(N)/np.log(2))):
                tmp = (np.power(2, n)-1) * self.K * self.q
                for j in range(self.q*(i-1)+1, self.q*i+1):
                    self.T[i-1].append(tmp + j)

        # initialize the models
        for m in self.forced_sample_model:
            m.fit([np.zeros(self.d)], [0])

        # start training
        for i in tqdm(range(len(labels))):
            self.update(data[i,:], labels[i])
        
    def update(self, features, l):
        self.sample_counter += 1
        choose_action = None
        # check to see if sample is in the forced sample sets
        for i in range(len(self.T)):
            if self.sample_counter in self.T[i]:
                choose_action = i
        
        # 
        
        choose_action = self._evaluate_datum(features)
        # observe reward r in {-1, 0}, turn it into {0, 1} for the algorithm
        # update A
        if l == choose_action:
            r = 1
        else:
            r = 0
            self.cumu_regret -= 1 # regret minus 1

        self.A[choose_action] += np.outer(features, features)
        self.b[choose_action] += features * r

        self.A_inv[choose_action] = np.linalg.inv(self.A[choose_action])
        self.theta[choose_action] = self.A_inv[choose_action].dot(self.b[choose_action])
        
        self.regret.append(self.cumu_regret)
        self.error_rate.append(-self.cumu_regret/self.sample_counter)

    
    def get_regret(self):
        return self.regret
    
    def get_error_rate(self):
        return self.error_rate
            
    def evaluate(self, data):
        """
        Given a data (NxM) input, return the corresponding dose

        returns a list (Nx1) of labels
        """
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.theta = [a_inv.dot(b) for a_inv, b in zip(self.A_inv, self.b)]
        
        labels = np.zeros(len(data))
        for i in range(len(data)):
            labels[i] = self._evaluate_datum(data[i])
        return labels
        
    def _evaluate_datum(self, features):
        
        p = np.zeros(self.K)
        for i in range(len(p)):
            tmp = features.T.dot(self.A_inv[i]).dot(features)
            p[i] = self.theta[i].dot(features) + self.alpha * np.sqrt(tmp)
         
        choose_action = np.argmax(p)
        return choose_action


if __name__ == '__main__':
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]
    
    ALPHA = 0.1

    lasso_bandit = Lasso_Bandit(alpha = ALPHA)
