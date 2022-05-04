import numpy as np
from .Utils import transform_coord_to_hex


class NeighbourhoodFunction:
    def __init__(self, neighbourhood_scaler=1):
        self.neighbourhood_scaler = neighbourhood_scaler
        self.is_hexagon = None
        
    def value(self, BMU, point, epoch):
        raise NotImplemented 

class NeighbourhoodGaussian(NeighbourhoodFunction):
    def __init__(self, neighbourhood_scaler=1):
        super().__init__(neighbourhood_scaler)
        
    def value(self, BMU, point, epoch):
        distance = None
        if not self.is_hexagon:
            distance = np.sqrt(np.sum(np.square(np.array(BMU) - np.array(point)))) * self.neighbourhood_scaler
        else:
            BMU1 = transform_coord_to_hex(BMU)
            point1 = transform_coord_to_hex(point)
            distance = np.sqrt(np.sum(np.square(np.array(BMU1) - np.array(point1))))
            
        return np.exp(
                -1 * (epoch * distance)**2
        ) 
        
        
class NeighbourhoodMexicanHat(NeighbourhoodFunction):
    def __init__(self, neighbourhood_scaler=1):
        super().__init__(neighbourhood_scaler)
        
    def value(self, BMU, point, epoch):
        distance = None
        if not self.is_hexagon:
            distance = np.sqrt(np.sum(np.square(np.array(BMU) - np.array(point)))) * self.neighbourhood_scaler
        else:
            BMU1 = transform_coord_to_hex(BMU)
            point1 = transform_coord_to_hex(point)
            distance = np.sqrt(np.sum(np.square(np.array(BMU1) - np.array(point1))))
            
        distance_epoch = (distance * epoch) ** 2
        return (2 - 4 * distance_epoch) * np.exp(-1 * distance_epoch)
