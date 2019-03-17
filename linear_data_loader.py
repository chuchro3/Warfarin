# Loads in the warfarin data file from './data/warfarin.csv'

import csv
import numpy as np
import sys

from collections import defaultdict
from data_loader import get_data

WARFARIN_FILE_PATH = './data/warfarin.csv'
DATA_COLUMNS = 65
LABEL_COLUMN = 'Therapeutic Dose of Warfarin'
 #FLOAT_LABELS = ['Age', 'Height (cm)', 'Weight (kg)', 'INR on Reported Therapeutic Dose of Warfarin']
FLOAT_LABELS = ['Height (cm)', 'Weight (kg)', 'INR on Reported Therapeutic Dose of Warfarin']
IGNORE_LABELS = ['Comorbidities', 'Medications']

# returns:
# - data (n x m)
# - labels (n x 1)
# - columns_dict (size m, keyed by string from warfarin data)
# - values_dict (size m, keyed by column name for data. each entry is a dict from data value string -> integer)
def get_data_linear():
    data, labels, columns_dict, values_dict = get_data()

    NUM_COLS = 0
    index_labels = {}
    for k in columns_dict:
        index_labels[columns_dict[k]] = k
        if k in FLOAT_LABELS:
            NUM_COLS += 1
        elif k not in IGNORE_LABELS:
            NUM_COLS += len(values_dict[k].items())
        print(k, ': ', NUM_COLS)

    # DANGEROUS: RAN ONCE TO FIGURE OUT SHAPE.
    NUM_ROWS = 5528
    # do not change the above unless shape of data changes.
    linearized_data = np.zeros((NUM_ROWS, NUM_COLS))
    print ("SHAPE OF DATA:", linearized_data.shape)
    for i, d in enumerate(data):
        write_index = 0
        for j, val in enumerate(d):
            if index_labels[j] in IGNORE_LABELS:
                continue
            elif index_labels[j] in FLOAT_LABELS:
                linearized_data[i,write_index] = val
                write_index += 1
            else:
                assert val == int(val), 'Value must be a value index'
                linearized_data[i,write_index + int(val)] = 1
                write_index += len(values_dict[index_labels[j]].items())
        #print("LEN:", write_index)
        assert write_index == NUM_COLS
    return linearized_data, labels

linearized_data, labels = get_data_linear()
