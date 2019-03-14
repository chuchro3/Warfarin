from baseline import Baseline

class Lin_UCB(Baseline):

    def __init__(self, columns_dict, values_dict, alpha, delta):
        super().__init__(columns_dict, values_dict)
        self.alpha = alpha
        self.delta = delta

    def __str__(self):
        return "Linear UCB (no fancy stuff)"

    def evaluate_datum(self, datum):
        return 0.0
