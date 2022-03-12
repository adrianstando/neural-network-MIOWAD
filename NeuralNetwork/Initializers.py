import numpy as np


class Initializer:
    def initialize(self, size):
        raise NotImplementedError


class RandomNormalInitializer(Initializer):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def initialize(self, size):
        return np.random.normal(self.mean, self.std, size)


class RandomUniformInitializer(Initializer):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def initialize(self, size):
        return np.random.uniform(self.low, self.high, size)


class XavierInitializer(Initializer):
    def __init__(self, dim_1, dim_2):
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def initialize(self, size):
        return np.random.uniform(-1, 1, size) * \
               (6 / (self.dim_1 + self.dim_2)) ** (1/2)
