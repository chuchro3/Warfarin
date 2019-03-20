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


# Weights can be found in 'data/appx.pdf' section 1f
class Warfarin_Pharmacogenetic_Dose(Baseline):

    def __str__(self):
        return "Warfarin Pharmacogenetic Dose"

    def _get_enzyme_inducer_status(self, datum):
        status = False
        status |= datum[self.columns_dict['Carbamazepine (Tegretol)']] == self.values_dict['Carbamazepine (Tegretol)']['1']
        status |= datum[self.columns_dict['Phenytoin (Dilantin)']] == self.values_dict['Phenytoin (Dilantin)']['1']
        status |= datum[self.columns_dict['Rifampin or Rifampicin']] == self.values_dict['Rifampin or Rifampicin']['1']
        return status

    # FOR MISSING WEIGHT/HEIGHT: use avg.
    def evaluate_datum(self, datum):
        dose = 5.6044
        dose -= 0.2614 * datum[self.columns_dict['Age']]
        dose += 0.0087 * datum[self.columns_dict['Height (cm)']]
        dose += 0.0128 * datum[self.columns_dict['Weight (kg)']]
        vk_gene = 'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'
        dose -= 0.8677 * datum[self.columns_dict[vk_gene]] == self.values_dict[vk_gene]['A/G']
        dose -= 1.6974 * datum[self.columns_dict[vk_gene]] == self.values_dict[vk_gene]['A/A']
        dose -= 0.4854 * datum[self.columns_dict[vk_gene]] == self.values_dict[vk_gene]['NA']
        dose -= 0.5211 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['*1/*2']
        dose -= 0.9357 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['*1/*3']
        dose -= 1.0616 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['*2/*2'] 
        dose -= 1.9206 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['*2/*3'] 
        dose -= 2.3312 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['*3/*3'] 
        dose -= 0.2188 * datum[self.columns_dict['CYP2C9 consensus']] == self.values_dict['CYP2C9 consensus']['NA'] 
        dose -= 0.1092 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['Asian'])
        dose -= 0.2760 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['Black or African American'])
        dose -= 0.1032 * (datum[self.columns_dict['Race']] == self.values_dict['Race']['NA'])
        dose += 1.1816 * self._get_enzyme_inducer_status(datum) 
        #Enzyme inducer status
        dose -= 0.5503 * datum[self.columns_dict['Amiodarone (Cordarone)']]
        # dose calculated in appx.pdf states that it's the sqrt of weekly
        return dose ** 2




