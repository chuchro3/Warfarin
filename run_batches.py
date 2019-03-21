import linear_data_loader as ldl
import data_loader as dl
import numpy as np
import random
import util
from lin_ucb import Lin_UCB
from util import plot_error_rate
from util import plot_regret
from lasso_bandit import LASSO_BANDIT

from baseline import Fixed_Dose, Warfarin_Clinical_Dose, Warfarin_Pharmacogenetic_Dose

import pickle



NUM_BATCHES = 10
ALPHA = 0.1


DATA_MULTIPLIER = 1

def run_model():
    data, true_labels = ldl.get_data_linear()
    true_buckets = [util.bucket(t) for t in true_labels]

    data = np.tile(data, (DATA_MULTIPLIER,1))
    print("DATA SHAPE:", data.shape)
    true_buckets = np.tile(true_buckets, DATA_MULTIPLIER)

    # tuples of (batch_id, total regret, error while training, eval error, precision, recall)
    batch_results = []

    for T in range(NUM_BATCHES):
        model = Lin_UCB(ALPHA)
        #model = LASSO_BANDIT()
        if False:
            data, true_labels, columns_dict, values_dict = dl.get_data()
            true_buckets = [util.bucket(t) for t in true_labels]
        #model = Fixed_Dose(columns_dict, values_dict)
        #model = Warfarin_Clinical_Dose(columns_dict, values_dict)
        #model = Warfarin_Pharmacogenetic_Dose(columns_dict, values_dict)
        
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

        with open('batch/regret'+str(model) + batch_id, 'wb') as fp:
            pickle.dump(model.regret, fp)
        with open('batch/error'+str(model) + batch_id, 'wb') as fp:
            pickle.dump(model.error_rate, fp)
        
    return batch_results


batch_results = run_model()
print()
print("Batch Results:")
print(batch_results)
