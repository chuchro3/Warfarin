# Loads in the warfarin data file from './data/warfarin.csv'

import csv
import numpy as np
import sys

from collections import defaultdict

WARFARIN_FILE_PATH = './data/warfarin.csv'
DATA_COLUMNS = 65
LABEL_COLUMN = 'Therapeutic Dose of Warfarin'

# returns:
# - data (n x m)
# - labels (n x 1)
# - columns_dict (size m, keyed by string from warfarin data)
# - values_dict (size m, keyed by column index for data. each entry is a dict from data value string -> integer)
def get_data():
  with open(WARFARIN_FILE_PATH) as csv_file:
    reader = csv.reader(csv_file, skipinitialspace=True)

    rows_parsed = 0
    columns_dict = {}
    values_dict = defaultdict(lambda: {'NA':0})
    data = []

    labels = []
    label_index = -1
    for i,r in enumerate(reader):
        if i == 0:
          print("Columns are:", r)
          print()
          print("Total num of columns is (excluding ID):", len(r)-1)
          print()
          for i,c in enumerate(r[1:]):
            if c == LABEL_COLUMN:
              label_index = i 
            else:
              columns_dict[c] = len(columns_dict)
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
              labels.append(float(col_val))
              adjust = 1
            else:
              if col_val not in values_dict[i - adjust]:
                values_dict[i-adjust][col_val] = len(values_dict[i - adjust])
              row.append(values_dict[i - adjust][col_val])

          if add_data:
            data.append(row)

    data = np.array(data)
    labels = np.array(labels)
    print("Finished parsing", rows_parsed, "rows from", WARFARIN_FILE_PATH)
    print("Shape of data:", data.shape)
    print("Shape of labels:", labels.shape)
    print("values_dict:", values_dict.keys())
    return data, labels, columns_dict, values_dict


#print(get_data()[1])
get_data()
