# Loads in the warfarin data file from './data/warfarin.csv'

import csv
import numpy as np
import sys

from collections import defaultdict
from data_loader import get_data

WARFARIN_FILE_PATH = './data/warfarin.csv'
DATA_COLUMNS = 65
LABEL_COLUMN = 'Therapeutic Dose of Warfarin'

FLOAT_LABELS = ['Age', 'Height (cm)', 'Weight (kg)']
LABELS = []

# returns:
# - data (n x m)
# - labels (n x 1)
# - columns_dict (size m, keyed by string from warfarin data)
# - values_dict (size m, keyed by column name for data. each entry is a dict from data value string -> integer)
def get_data_linear():
    data, labels, columns_dict, values_dict = get_data()

    index_labels = {}
    for k in columns_dict:
        index_labels[columns_dict[k]] = k

    # DANGEROUS: RAN ONCE TO FIGURE OUT SHAPE.
    NUM_ROWS = 5528
    NUM_COLS = 4153
    # do not change the above unless shape of data changes.
    linearized_data = np.zeros((NUM_ROWS, NUM_COLS))
    for i, d in enumerate(data):
        write_index = 0
        for j, val in enumerate(d):
            if index_labels[j] in FLOAT_LABELS:
                linearized_data[i,write_index] = val
                write_index += 1
            else:
                assert val == int(val), 'Value must be a value index'
                linearized_data[i,write_index + int(val)] = 1
                write_index += len(values_dict[index_labels[j]].items())
        assert write_index == NUM_COLS
    return linearized_data, labels

