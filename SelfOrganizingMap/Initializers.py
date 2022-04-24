import numpy as np

class Initializer:
    def initialize(self, width, height, vector_in_size):
        raise NotImplementedError


class RandomUniformInitializer(Initializer):
    def __init__(self, low_lim=0, high_lim=1, seed=None):
        self.seed = seed
        self.low_lim = low_lim
        self.high_lim = high_lim
        
    def initialize(self, width, height, vector_in_size):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.low_lim, self.high_lim, (vector_in_size, height, width)) 