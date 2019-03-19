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

def evaluate_performance(labels, true_labels):
    
    acc = get_accuracy_bucketed(labels, true_labels)
    prec = get_bucket_precision(labels, true_labels)
    recall = get_bucket_recall(labels, true_labels)
    
    print("Accuracy: " + str(acc))
    print("Precision: " + str(prec))
    print("Recall: " + str(recall))
    return acc, prec, recall

def get_accuracy(labels, true_labels):
    corr = 0
    for i in range(len(labels)):
        if bucket(labels[i]) == bucket(true_labels[i]):
            corr += 1

    return corr / len(labels)

def get_bucket_precision(labels, true_labels):
    buckets = set()
    acc_dic = {}
    t_dic = {}
    for i in range(len(labels)):
        #l = bucket(labels[i])
        l = labels[i] 
        #lt = bucket(true_labels[i])
        lt = true_labels[i]
        buckets.add(l)
        buckets.add(lt)
        if l == lt:
            if l in acc_dic:
                acc_dic[lt] += 1
            else:
                acc_dic[lt] = 1
        if l in t_dic:
            t_dic[l] += 1
        else:
            t_dic[l] = 1
    acc = np.zeros(len(buckets))
    print(acc_dic)
    print(t_dic)
    for k in acc_dic:
        acc[k] = 1. * acc_dic[k] / t_dic[k]
    return acc


def get_bucket_recall(labels, true_labels):
    buckets = set()
    acc_dic = {}
    t_dic = {}
    for i in range(len(labels)):
        #l = bucket(labels[i])
        l = labels[i] 
        #lt = bucket(true_labels[i])
        lt = true_labels[i]
        buckets.add(l)
        buckets.add(lt)
        if l == lt:
            if l in acc_dic:
                acc_dic[lt] += 1
            else:
                acc_dic[lt] = 1
        if lt in t_dic:
            t_dic[lt] += 1
        else:
            t_dic[lt] = 1
    acc = np.zeros(len(buckets))
    print(acc_dic)
    print(t_dic)
    for k in acc_dic:
        acc[k] = 1. * acc_dic[k] / t_dic[k]
    return acc

def get_accuracy_bucketed(labels, true_labels):
    corr = 0
    for i in range(len(labels)):
        if labels[i] == true_labels[i]:
            corr += 1

    return corr / len(labels)
