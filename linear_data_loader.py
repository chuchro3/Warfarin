# Loads in the warfarin data file from './data/warfarin.csv'

import csv
import numpy as np
import sys

from collections import defaultdict
from data_loader import get_data

FLOAT_LABELS = [
    'Age',
    'Height (cm)',
    'Weight (kg)',
    'INR on Reported Therapeutic Dose of Warfarin',
]

IGNORE_LABELS = [
    'Medications',
    'Comorbidities'
]

DISEASE_LABEL = 'Comorbidities'

# switch to True if want to 
INCLUDE_DISEASES = False

# DO NOT ACCESS NUM_COLS BEFORE CALLING get_data_linear
NUM_COLS = 0
NUM_COLS_INITIALIZED = False


# returns:
# - linearized_data: one-hot encoding of everything except for the columns in
#                    FLOAT_LABELS and ignoring those in IGNORE_LABELS.
#                    data 
# - labels: unchanged from return of get_data(); just included for convenience
def get_data_linear():
    data, labels, columns_dict, values_dict = get_data()

    # preprocess the diseases
    disease_dict = {}
    key_to_disease_string = {}
    for k in values_dict[DISEASE_LABEL]:
        diseases = map(str.strip, k.split(';'))
        key_to_disease_string[values_dict[DISEASE_LABEL][k]] = k
        for d in diseases:
            if d not in disease_dict:
                disease_dict[d] = len(disease_dict)
    print("DISEASES:", len(disease_dict))
    # end disease preprocess

    # DANGEROUS: RAN ONCE TO FIGURE OUT NUM OF ROWS.
    NUM_ROWS = 5528
    global NUM_COLS
    global NUM_COLS_INITIALIZED
    if not NUM_COLS_INITIALIZED:
        for k in columns_dict:
            if k in FLOAT_LABELS:
                NUM_COLS += 1
            elif k not in IGNORE_LABELS:
                NUM_COLS += len(values_dict[k].items())
        NUM_COLS_INITIALIZED = True

        if INCLUDE_DISEASES:
            NUM_COLS += len(disease_dict)
    # do not change the above unless amount of data changes.


    index_labels = {}
    for k in columns_dict:
        index_labels[columns_dict[k]] = k

    linearized_data = np.zeros((NUM_ROWS, NUM_COLS))
    print ("SHAPE OF DATA:", linearized_data.shape)
    for i, d in enumerate(data):
        write_index = 0
        for j, val in enumerate(d):
            if index_labels[j] in FLOAT_LABELS:
                linearized_data[i,write_index] = val
                write_index += 1
            elif INCLUDE_DISEASES and index_labels[j] == DISEASE_LABEL:
                diseases = map(str.strip, key_to_disease_string[val].split(';'))
                for dis in diseases:
                    linearized_data[i, write_index + disease_dict[dis]] = 1
                write_index += len(disease_dict)
            elif index_labels[j] in IGNORE_LABELS:
                continue
            else:
                assert val == int(val), 'Value must be a value index'
                linearized_data[i,write_index + int(val)] = 1
                write_index += len(values_dict[index_labels[j]].items())
        assert write_index == NUM_COLS
    return linearized_data, labels

linearized_data, labels = get_data_linear()
