import linear_data_loader as ldl
import numpy as np
import util
from util import plot_error_rate
from util import plot_regret
from tqdm import tqdm

FEATURE_DIM = ldl.NUM_COLS
NUM_ACTIONS = 3

class Lin_UCB():
    def __init__(self, alpha, K=NUM_ACTIONS, d=FEATURE_DIM):
        self.alpha = alpha
        self.K = K
        self.d = d
        self.A = [np.identity(self.d) for k in range(K)]
        self.A_inv = [np.identity(self.d) for k in range(K)]
        self.b = [np.zeros(self.d) for k in range(K)]
        self.theta = None
        self.regret = []
        self.error_rate = []
        self.cumu_regret = 0
        self.sample_counter = 0

        # evaluating A_inv and theta moved here for efficiency.
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.theta = [a_inv.dot(b) for a_inv, b in zip(self.A_inv, self.b)]

    def __str__(self):
        return "Linear UCB"
    
    def train(self, data, labels):
        for i in tqdm(range(len(labels))):
            self.update(data[i,:], labels[i])
        
    def update(self, features, l):
        self.sample_counter += 1
        
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

    
def test_lin_ucb_full(data, true_buckets, alpha=0.1):
    lin_ucb = Lin_UCB(alpha = alpha)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on linear UCB: " + str(acc))


if __name__ == '__main__':
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]
    
    ALPHA = 0.1

    lin_ucb = Lin_UCB(alpha = ALPHA)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on linear UCB: " + str(acc))
    plot_regret(lin_ucb.regret, ALPHA)
    plot_error_rate(lin_ucb.error_rate, ALPHA)
