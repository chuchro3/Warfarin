# Loads in the warfarin data file from './data/warfarin.csv'

import csv
import numpy as np
import sys

from collections import defaultdict

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
def get_data():
  with open(WARFARIN_FILE_PATH) as csv_file:
    reader = csv.reader(csv_file, skipinitialspace=True)

    rows_parsed = 0
    columns_dict = {}
    values_dict = defaultdict(lambda: {'NA':0})
    data = []

    labels = []
    label_index = -1
    ignore_columns_past_index = float('inf')

    weights = []
    heights = []

    for i,r in enumerate(reader):
        if i == 0:
          for i,c in enumerate(r[1:]):
            if c == LABEL_COLUMN:
              label_index = i 
            elif c == '':
              ignore_columns_past_index = min(ignore_columns_past_index, i)
            else:
              columns_dict[c] = len(columns_dict)
              LABELS.append(c)
        elif r[0] != '':  # check that subject ID is present
          rows_parsed += 1
          row = []
          adjust = 0
          add_data = True
          for i,col_val in enumerate(r[1:]):
            if i == label_index:
              if col_val == 'NA':  # toss out this sample
                add_data = False
                break
              label = float(col_val)
              adjust = 1
            elif i < ignore_columns_past_index:
              if LABELS[i - adjust] in FLOAT_LABELS:
                if LABELS[i - adjust] == 'Age':
                    try:
                        col_val = float(col_val[0])
                    except:
                        # age could've been NA or a random date
                        col_val = 0
                else:
                    try:
                        col_val = float(col_val)
                    except:
                        col_val = 0
                    if col_val != 0:
                        if LABELS[i - adjust] == 'Height (cm)':
                            heights.append(col_val)
                        elif LABELS[i - adjust] == 'Weight (kg)':
                            weights.append(col_val)
                        else:
                            raise Exception('Should not happen')
                row.append(col_val)
              else:
                  label_class = LABELS[i - adjust]
                  if col_val not in values_dict[label_class]:
                    values_dict[label_class][col_val] = len(values_dict[label_class])
                  row.append(values_dict[label_class][col_val])

          if add_data:
            data.append(row)
            labels.append(label)

    assert len(values_dict.keys()) + len(FLOAT_LABELS) == len(columns_dict.keys()), "length of non-float values and total columns dicts should match"

    data = np.array(data)
    labels = np.array(labels)
    print("Finished parsing", rows_parsed, "rows from", WARFARIN_FILE_PATH)
    # print("Shape of data:", data.shape)
    # print("Shape of labels:", labels.shape)

    avg_height = np.mean(heights)
    avg_weight = np.mean(heights)
    for i in range(data.shape[0]):
        if data[i][columns_dict['Height (cm)']] == 0:
            data[i][columns_dict['Height (cm)']] = avg_height
        if data[i][columns_dict['Weight (kg)']] == 0:
            data[i][columns_dict['Weight (kg)']] = avg_weight

    return data, labels, columns_dict, values_dict


#print(get_data()[1])
data, labels, columns_dict, values_dict = get_data()
