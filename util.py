import numpy as np

def bucket(dose):
    """
    Returns the proper bucket given a dose
    0 :- dose < 21
    1 :- 21 <= dose <= 49
    2 :- dose > 49
    """
    if dose < 21:
        return 0
    if dose > 49:
        return 2
    return 1


def get_accuracy(labels, true_labels):
    corr = 0
    for i in range(len(labels)):
        if bucket(labels[i]) == bucket(true_labels[i]):
            corr += 1

    return corr / len(labels)


def get_accuracy_bucketed(labels, true_labels):
    corr = 0
    for i in range(len(labels)):
        if labels[i] == true_labels[i]:
            corr += 1

    return corr / len(labels)
