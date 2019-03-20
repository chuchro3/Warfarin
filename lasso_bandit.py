import linear_data_loader as ldl
import numpy as np
import util
from tqdm import tqdm
from numpy import linalg as LA
from scipy import optimize
from sklearn import linear_model
import matplotlib.pyplot as plt
import datetime

FORCED_SAMPLES = 50
FEATURE_DIM = ldl.NUM_COLS
NUM_ACTIONS = 3
LAMBDA = 0.1

UPDATE_EVERY_X_ITERS = 1


class LASSO_BANDIT():
    def _predict_reward(self, arm, features):
        return self.model[arm].predict([features])[0]

    # l is lambda (regularization weight)
    def _update_model(self, arm, X, Y, l):
        key = (arm, X.shape[0] / UPDATE_EVERY_X_ITERS)
        if key in self.mem:
            return # don't perform update yet
        else:
            #print("not in cache", key)
            pass

        self.model[arm].set_params(alpha = l)
        self.model[arm].fit(X, Y)
        self.mem.add(key)

    def __init__(self, K=NUM_ACTIONS, d=FEATURE_DIM):
        self.K = K
        self.d = d
        self.regret = []
        self.error_rate = []
        self.cumu_regret = 0

        # LASSO bandit specific params
        self.T_i = [set(range(arm*FORCED_SAMPLES, (arm+1)*FORCED_SAMPLES)) for arm in range(self.K)]
        self.S_i = [[] for arm in range(self.K)]
        self.Y_i = [[] for arm in range(self.K)]
        self.timestep_t = 0
        self.lambda_0 = LAMBDA

        # set for what data sizes have been fit for each arm. contains no values.
        self.mem = set()
        # TODO: can train even faster with larger tolerance. 'tol' param for Lasso
        self.model = [linear_model.Lasso(alpha=self.lambda_0, warm_start=True) for arm in range(self.K)]


    def __str__(self):
        return "LASSO Bandit"
    
    def train(self, data, labels):
        for i in tqdm(range(len(labels))):
            self.update(data[i,:], labels[i])
        
    def update(self, features, l):
        self.timestep_t += 1
        action_t = -1

        # exploration if iter is inside some T_i
        for arm in range(self.K):
            if self.timestep_t in self.T_i[arm]:
                action_t = arm
                break

        # else, exploit based on best arm.
        if action_t == -1:
            best_reward = float('-inf')
            for arm in range(self.K):
                predicted_arm_reward = self._predict_reward(arm, features)
                if predicted_arm_reward > best_reward:
                    action_t = arm
                    best_reward = predicted_arm_reward

        assert action_t != -1, "No arm was selected..."

        # add the arm to the observations
        self.S_i[action_t].append(features)
        # update lambda
        self.lambda_t = self.lambda_0 * np.sqrt(
                (np.log((self.timestep_t)) + np.log(self.d)) / self.timestep_t)

        # play the arm, and observe the reward.
        self.Y_i[action_t].append(0 if action_t == l else -1)

        # update the model for the arm that was pulled
        self._update_model(action_t,
                           np.array(self.S_i[action_t]),
                           np.array(self.Y_i[action_t]),
                           self.lambda_t)

        # metrics collection
        self.cumu_regret += (0 if action_t == l else -1)
        self.regret.append(self.cumu_regret)
        self.error_rate.append(-self.cumu_regret/self.timestep_t)

        return action_t

    
    def get_regret(self):
        return self.regret
    
    def get_error_rate(self):
        return self.error_rate
            
    def evaluate(self, data):
        """
        Given a data (NxM) input, return the corresponding dose

        returns a list (Nx1) of labels
        """
        labels = np.zeros(len(data))
        for i in range(len(data)):
            labels[i] = self._evaluate_datum(data[i])
        return labels
        
    def _evaluate_datum(self, features):
        action_t = -1
        best_reward = float('-inf')
        for arm in range(self.K):
            predicted_arm_reward = self._predict_reward(arm, features)
            if predicted_arm_reward > best_reward:
                action_t = arm
                best_reward = predicted_arm_reward

        assert action_t != -1, "[eval datum] No arm was selected..."
        return action_t

    
# probably do not run this for decent results (below data isn't randomized)
# execute 'run_batches.py' instead on the lasso bandit model.
if __name__ == '__main__':
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]
    
    lasso_bandit = LASSO_BANDIT()
    lasso_bandit.train(data[:2000], true_buckets[:2000])
    pred_buckets = lasso_bandit.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on LASSO bandit: " + str(acc))
    #plot_regret(lasso_bandit.regret, ALPHA)
    #plot_error_rate(lasso_bandit.error_rate, ALPHA)
