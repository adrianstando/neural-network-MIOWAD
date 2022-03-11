import numpy as np


def generate_vectorized_function(func):
    f = np.vectorize(func)
    return lambda x: f(x)
