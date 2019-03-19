import linear_data_loader as ldl
import numpy as np
import random
import util
from lin_ucb import Lin_UCB
from util import plot_error_rate
from util import plot_regret
from lasso_bandit import LASSO_BANDIT


NUM_BATCHES = 10
ALPHA = 0.1


def run_model(model):
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]

    # tuples of (batch_id, total regret, error while training, eval error, precision, recall)
    batch_results = []

    for T in range(NUM_BATCHES):
        batch_id = str(random.randint(100000, 999999))
        print()
        print("Start Batch: ", batch_id)

        zipped_data = list(zip(data, true_buckets))
        random.shuffle(zipped_data)
        data, true_buckets = zip(*zipped_data)
        data = np.array(data)

        model.train(data, true_buckets)
        pred_buckets = model.evaluate(data)
        print(batch_id, "Performance on " + str(model))
        acc, precision, recall = util.evaluate_performance(pred_buckets, true_buckets)
        print("\tAccuracy:", acc)
        print("\tPrecision:", precision)
        print("\tRecall:", recall)
        

        plot_regret(model.regret, ALPHA, batch_id)
        plot_error_rate(model.error_rate, ALPHA, batch_id)

        batch_results.append(
            (batch_id,
             model.get_regret()[-1],
             model.get_error_rate()[-1],
             1 - acc,
             precision,
             recall))
    return batch_results

#model = Lin_UCB(ALPHA)
model = LASSO_BANDIT()

batch_results = run_model(model)
print()
print("Batch Results:")
print(batch_results)
