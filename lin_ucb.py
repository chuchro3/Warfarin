from baseline import Baseline
import linear_data_loader as ldl
import numpy as np
import util
from tqdm import tqdm

FEATURE_DIM = 297
NUM_ACTIONS = 3

class Lin_UCB():
    def __init__(self, alpha, K=NUM_ACTIONS, d=FEATURE_DIM):
        self.alpha = alpha
        self.K = K
        self.d = d
        self.A = [np.identity(self.d)] * K 
        self.A_inv = [np.identity(self.d)] * K 
        self.b = [np.zeros(self.d)] * K 
        self.theta = None

    def __str__(self):
        return "Linear UCB"
    
    def train(self, data, labels):
        for i in tqdm(range(len(labels))):
            self.update(data[i,:], labels[i])
        
    def update(self, features, l):
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.theta = [a_inv.dot(b) for a_inv, b in zip(self.A_inv, self.b)]
        choose_action = self._evaluate_datum(features)
        # observe reward r in {-1, 0}, turn it into {0, 1} for the algorithm
        # update A
        if l == choose_action:
            r = 1
        else:
            r = 0
        self.A[choose_action] += np.outer(features, features)
        self.b[choose_action] += features * r
    
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
    
def test_lin_ucb_full(data, true_buckets, alpha=0.1):
    lin_ucb = Lin_UCB(alpha = alpha)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on linear UCB: " + str(acc))

if __name__ == '__main__':
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]
    
    lin_ucb = Lin_UCB(alpha = 0.1)
    lin_ucb.train(data[1:3,:], true_labels[1:3])
    pred_buckets = lin_ucb.evaluate(data[5:10,])
    
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets[5:10])
    print("accuracy on linear UCB: " + str(acc))
    
    # test_lin_ucb_full(data, true_buckets, alpha=0.1)
    lin_ucb = Lin_UCB(alpha = 0.1)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on linear UCB: " + str(acc))
