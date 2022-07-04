import numpy as np


class Acquisiton:
    def __init__(self, X, k):
        self.X = X
        self.k = k

    def random(self):
        return np.random.choice(self.X, int(self.k * len(self.X)))
