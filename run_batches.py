import linear_data_loader as ldl
import numpy as np
import random
import util
from lin_ucb import Lin_UCB
from lin_ucb import plot_error_rate
from lin_ucb import plot_regret


NUM_BATCHES = 10
ALPHA = 0.1

data, true_labels = ldl.get_data_linear()
true_buckets = [util.bucket(t) for t in true_labels]

# tuples of (batch_id, total regret, error while training, eval error)
batch_results = []

for T in range(NUM_BATCHES):
    batch_id = str(random.randint(100000, 999999))
    print()
    print("Start Batch: ", batch_id)

    zipped_data = list(zip(data, true_buckets))
    random.shuffle(zipped_data)
    data, true_buckets = zip(*zipped_data)
    data = np.array(data)

    lin_ucb = Lin_UCB(alpha = ALPHA)
    lin_ucb.train(data, true_buckets)
    pred_buckets = lin_ucb.evaluate(data)
    acc = util.get_accuracy_bucketed(pred_buckets, true_buckets)
    print(batch_id, "accuracy on linear UCB: " + str(acc))
    bucket_acc = util.get_bucket_accuracy(pred_buckets, true_buckets)
    print(batch_id, "bucket accuracy on linear UCB: " + str(bucket_acc))

    plot_regret(lin_ucb.regret, ALPHA, batch_id)
    plot_error_rate(lin_ucb.error_rate, ALPHA, batch_id)

    batch_results.append(
        (batch_id,
         lin_ucb.get_regret()[-1],
         lin_ucb.get_error_rate()[-1],
         1 - acc))

print()
print("Batch Results:")
print(batch_results)
