import numpy as np


class ActivationFunction:
    def __init__(self):
        self.is_vectorized = None

    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented


class SigmoidFunction(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.is_vectorized = False

    def function(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))


class LinearFunction(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.is_vectorized = False

    def function(self, x):
        return x

    def derivative(self, x):
        return 1


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.is_vectorized = False

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class SoftmaxAsLastLayerWithCrossEntropy(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.is_vectorized = True

    def function(self, x):
        # exps = np.exp(x)
        # return exps / np.sum(exps)

        # stable version
        if x.ndim == 1:
            shiftx = x - np.max(x)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)
        else:
            shiftx = np.apply_along_axis(lambda y: y - np.max(y), 1, x)
            exps = np.exp(shiftx)
            return np.apply_along_axis(lambda y: y / np.sum(y), 1, exps)

    def derivative(self, x):
        # if softmax is the last layer for problem with cross-entropy loss function,
        # code can be simplified to this
        # source: http://neuralnetworksanddeeplearning.com/chap3.html#problems_805405
        return np.ones_like(x)

    
class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.is_vectorized = False

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        # assuming that derivative in x=0 is 0
        return (np.maximum(0, x) > 0).astype(int)
