import numpy as np


class NeighbourhoodFunction:
    def value(self, BMU, point, epoch):
        raise NotImplemented 

class NeighbourhoodGaussian(NeighbourhoodFunction):
    def value(self, BMU, point, epoch):
        distance = np.sqrt(np.sum(np.square(np.array(BMU) - np.array(point))))
        return np.exp(
                -1 * (epoch * distance)**2
        ) 
        
        
class NeighbourhoodMexicanHat(NeighbourhoodFunction):
    def value(self, BMU, point, epoch):
        distance = np.sqrt(np.sum(np.square(np.array(BMU) - np.array(point))))
        e_2 = epoch**2
        d_2 = distance**2
        return (2 * e_2 - 4 * e_2**2 * d_2) * np.exp(-1 * e_2 * d_2)