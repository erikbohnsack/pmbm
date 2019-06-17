class GlobalHypothesis:
    def __init__(self, weight, hypothesis):
        self.weight = weight
        self.hypothesis = hypothesis

    def __repr__(self):
        return '<Global Hypothesis: \n  weight: {} \n  hypothesis: {} >\n'.format(
            self.weight, self.hypothesis)
