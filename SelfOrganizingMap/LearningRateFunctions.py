import numpy as np


class LearningRateFunction:
    def value(self, epoch):
        raise NotImplemented 

class StandardFunction(LearningRateFunction):
    def __init__(self, lambda_param):
        self.lambda_param = lambda_param
    
    def value(self, epoch):
        return np.exp(
                -1 * epoch * (1/self.lambda_param)
        ) 
         
