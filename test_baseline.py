import numpy as np
import data_loader as dl
import util
from baseline import Fixed_Dose

def test_dummy_baseline(alg):
    n = 3
    m = 2
    true_labels = [0,30,57]
    data = np.zeros((n,m))
    
    alg.train(data, true_labels)

    labels = alg.evaluate(data)

    acc = util.get_accuracy(labels, true_labels)
    print("accuracy on dummy data with " + str(alg) + ": " + str(acc))

def test_data_baseline(alg, data, true_labels):

    alg.train(data, true_labels)
    
    labels = alg.evaluate(data)

    acc = util.get_accuracy(labels, true_labels)
    print("accuracy on data with " + str(alg) + ": " + str(acc))

if __name__ == '__main__':

    data, true_labels, columns_dict, values_dict = dl.get_data()
    
    fixed = Fixed_Dose(columns_dict, values_dict)
    
    test_dummy_baseline(fixed)
    test_data_baseline(fixed, data, true_labels)
