import numpy as np


# http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization
class Regularization:
    def function(self, weights):
        raise NotImplemented

    def derivative(self, weights):
        raise NotImplemented


class RegularizationNone(Regularization):
    def function(self, weights):
        return 0
    
    def derivative(self, weights):
        return [np.zeros(w.shape) for w in weights]


class RegularizationL1(Regularization):
    def __init__(self, reg_param = 0.1):
        self.reg_param = reg_param
        
    def function(self, weights):
        return self.reg_param * np.sum(np.array([np.sum(np.abs(w)) for w in weights]))
    
    def derivative(self, weights):
        return [self.reg_param * np.sign(w) for w in weights]
    
    
class RegularizationL2(Regularization):
    def __init__(self, reg_param = 0.1):
        self.reg_param = reg_param
        
    def function(self, weights):
        return self.reg_param * np.sum(np.array([np.sum(np.square(w)) for w in weights])) 
    
    def derivative(self, weights):
        return [2 * self.reg_param * w for w in weights] 
