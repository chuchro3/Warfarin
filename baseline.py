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

# Weights can be found in 'data/appx.pdf' section 1f
class Warfarin_Clinical_Dose(Baseline):

    def __str__(self):
        return "Warfarin Clinical Dose"

    def _get_enzyme_inducer_status(self, datum):
        status = False
        status |= datum[self.columns_dict['Carbamazepine (Tegretol)']] == self.values_dict['Carbamazepine (Tegretol)']['1']
        status |= datum[self.columns_dict['Phenytoin (Dilantin)']] == self.values_dict['Phenytoin (Dilantin)']['1']
        status |= datum[self.columns_dict['Rifampin or Rifampicin']] == self.values_dict['Rifampin or Rifampicin']['1']
        return status

    # FOR MISSING WEIGHT/HEIGHT: use avg.
    def evaluate_datum(self, datum):
        dose = 4.0376
        dose -= 0.2546 * datum[self.columns_dict['Age']]
        dose += 0.0118 * datum[self.columns_dict['Height (cm)']]
        dose += 0.0134 * datum[self.columns_dict['Weight (kg)']]
        dose -= 0.6752 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['Asian'])
        dose += 0.4060 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['Black or African American'])
        dose += 0.0443 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['NA'])
        dose += 0.0443 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['Unknown'])
        dose += 1.2799 * self._get_enzyme_inducer_status(datum)
        dose -= 0.5695 * (datum[self.columns_dict['Amiodarone (Cordarone)']] == self.values_dict['Amiodarone (Cordarone)']['1'])
        # dose calculated in appx.pdf states that it's the sqrt of weekly
        return dose ** 2






