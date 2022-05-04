import numpy as np

def transform_coord_to_hex(point):
    p = [0, 0]
    if point[1] % 2 == 0:
        p[0] = point[0] * np.sqrt(3)
    else:
        p[0] = np.sqrt(3) / 2 + np.sqrt(3) * point[0]

    p[1] = point[1] * 3/2
    
    return (p[0], p[1])
