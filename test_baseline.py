import numpy as np
import util
from baseline import Fixed_Dose

def test_dummy_baseline():
    n = 3
    m = 2
    true_labels = [0,1,2]
    data = np.zeros((n,m))
    
    fixed = Fixed_Dose() 

    labels = np.zeros(len(true_labels))
    for i in range(len(data)):
        datum = data[i]
        labels[i] = fixed.evaluate(datum)

    acc = util.get_accuracy(labels, true_labels)
    print(acc)

if __name__ == '__main__':
    test_dummy_baseline()
