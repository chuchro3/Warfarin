import numpy as np
import util

class Baseline(object):

    def __init__(self, columns_dict=None, values_dict=None):
        self.columns_dict = columns_dict
        self.values_dict = values_dict


    def train(self, data, labels):
        pass 

    def evaluate(self, data):
        """
        Given a data (NxM) input, return the corresponding dose

        returns a list (Nx1) of labels
        """
        labels = np.zeros(len(data))
        for i in range(len(data)):
            labels[i] = self.evaluate_datum(data[i])
        return labels

    def evaluate_datum(self, datum):
        """
        Given a data input, return the corresponding dose
        """
        pass


    
class Fixed_Dose(Baseline):

    def __str__(self):
        return "Fixed"

    def evaluate_datum(self, datum):
        return 35 

