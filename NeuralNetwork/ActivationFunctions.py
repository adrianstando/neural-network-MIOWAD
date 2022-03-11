import numpy as np


class ActivationFunction:
    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented


class SigmoidFunction(ActivationFunction):
    def function(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))


class LinearFunction(ActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x):
        return 1


class Tanh(ActivationFunction):
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))
