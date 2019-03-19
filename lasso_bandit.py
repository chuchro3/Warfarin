import linear_data_loader as ldl
import numpy as np
import util
from tqdm import tqdm
from numpy import linalg as LA
from scipy import optimize
from sklearn import linear_model
import matplotlib.pyplot as plt
import datetime

FEATURE_DIM = ldl.NUM_COLS
NUM_ACTIONS = 3
LAMBDA = 0.01


class LASSO_BANDIT():
    def _compute_beta_hat(self, arm, X, Y, l=0.1):
        # l is lambda
        """
        def func_beta_hat(beta):
            return (LA.norm(Y - np.dot(X,beta))**2) / X.shape[0] + l * LA.norm(beta, 1)

        initial = np.zeros(X.shape[1])
        return optimize.minimize(func_beta_hat, initial)
        """
        if (arm, X.shape) in self.mem:
            return self.mem[(arm, X.shape)]
        model = linear_model.Lasso(alpha=l)
        model.fit(X, Y)
        self.mem[(arm, X.shape)] = model.coef_
        return model.coef_


    def __init__(self, K=NUM_ACTIONS, d=FEATURE_DIM):
        """
        self.alpha = alpha
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
        """
        self.K = K
        self.d = d
        self.regret = []
        self.error_rate = []
        self.cumu_regret = 0
        # LASSO bandit specific params
        self.T_i = [set(range(arm*200, (arm+1)*200)) for arm in range(self.K)]
        self.S_i = [[] for arm in range(self.K)]
        self.Y_i = [[] for arm in range(self.K)]
        self.timestep_t = 0
        self.lambda_0 = LAMBDA
        # cache for compute beta hat
        self.mem = {}


    def __str__(self):
        return "Linear UCB"
    
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
                predicted_arm_reward = np.dot(features.T, self._compute_beta_hat(arm, np.array(self.S_i[arm]), np.array(self.Y_i[arm]), self.lambda_t))
                if predicted_arm_reward > best_reward:
                    action_t = arm
                    best_reward = predicted_arm_reward

        assert action_t != -1, "No arm ws selected..."

        # add the arm to the observations
        self.S_i[action_t].append(features)
        # update lambda
        self.lambda_t = self.lambda_0 * np.sqrt((np.log((self.timestep_t)) + np.log(self.d)) / self.timestep_t)

        # play the arm, and observe the reward.
        self.Y_i[action_t].append(0 if action_t == l else -1)

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
            predicted_arm_reward = np.dot(features.T, self._compute_beta_hat(arm, np.array(self.S_i[arm]), np.array(self.Y_i[arm]), self.lambda_t))
            if predicted_arm_reward > best_reward:
                action_t = arm
                best_reward = predicted_arm_reward

        assert action_t != -1, "[eval datum] No arm ws selected..."
        return action_t

    
def test_lin_ucb_full(data, true_buckets, alpha=0.1):
    lin_ucb = Lin_UCB(alpha = alpha)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on linear UCB: " + str(acc))

def plot_regret(regrets, alpha, suffix=''):
    plt.clf()
    plt.title('Regret - ' + str(alpha) + ' - ' + datetime.datetime.now().strftime("%D - %H:%M:%S"))
    plt.xlabel('Samples Seen')
    plt.ylabel('Regret')
    plt.plot(range(1, 1+ len(regrets)), regrets)
    if suffix == '':
        plt.savefig('plots/regret'+str(alpha).replace('.','_')+ '_' + datetime.datetime.now().strftime('%s'))
    else:
        plt.savefig('plots/regret'+ '_' + suffix)

def plot_error_rate(error_rates, alpha, suffix=''):
    plt.clf()
    plt.title('Error Rate- ' + str(alpha) + ' - ' + datetime.datetime.now().strftime("%D - %H:%M:%S"))
    plt.xlabel('Samples Seen')
    plt.ylabel('Cumulative Error Rate')
    plt.plot(range(1, 1+ len(error_rates)), error_rates)
    if suffix == '':
        plt.savefig('plots/error'+str(alpha).replace('.','_')+ '_' +datetime.datetime.now().strftime('%s'))
    else:
        plt.savefig('plots/error'+ '_' + suffix)

if __name__ == '__main__':
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]
    
    lasso_bandit = LASSO_BANDIT()
    lasso_bandit.train(data[:3000], true_buckets[:3000])
    pred_buckets = lasso_bandit.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print("accuracy on LASSO bandit: " + str(acc))
    #plot_regret(lasso_bandit.regret, ALPHA)
    #plot_error_rate(lasso_bandit.error_rate, ALPHA)
