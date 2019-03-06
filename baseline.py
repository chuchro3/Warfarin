import util

class Baseline(object):

    def __init__(self):
        pass


    def train(self, data):
        pass

    def evaluate(self, datum):
        """
        Given a data input, return the corresponding dose
        """
        pass



    
class Fixed_Dose(Baseline):

    def evaluate(self, datum):
        return util.bucket(35)
