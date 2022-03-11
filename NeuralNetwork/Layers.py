import numpy as np
from .ActivationFunctions import ActivationFunction
from .Utils import generate_vectorized_function
from .Initializers import Initializer, RandomUniformInitializer


class Layer:
    def forward(self, vector_in):
        raise NotImplemented

    def set_weights(self, weights):
        raise NotImplemented

    def set_biases(self, biases):
        raise NotImplemented

    def weighted_input(self, vector_in):
        raise NotImplemented

    def summary(self):
        raise NotImplemented


class DenseNetLayer(Layer):
    def __init__(self, input_size, number_of_neurons, activation_function: ActivationFunction, initializer: Initializer):
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.weights = initializer.initialize((input_size, number_of_neurons))
        self.biases = initializer.initialize(number_of_neurons)
        self.initializer = initializer
        self.activation_function = generate_vectorized_function(activation_function.function)
        self.activation_function_derivative = generate_vectorized_function(activation_function.derivative)

    def forward(self, vector_in):
        y = vector_in @ self.weights
        y = y + self.biases
        return self.activation_function(y)

    def weighted_input(self, vector_in):
        y = vector_in @ self.weights
        y = y + self.biases
        return y

    def set_weights(self, weights):
        if weights.shape[1] != self.number_of_neurons or \
                weights.shape[0] != self.input_size or \
                weights.ndim != 2:
            print("Wrong weights size!")
        else:
            self.weights = weights

    def set_biases(self, biases):
        if biases.shape[0] != self.number_of_neurons or \
                biases.ndim != 1:
            print("Wrong bias size!")
        else:
            self.biases = biases

    def summary(self):
        print(f"Input size: {self.input_size}")
        print(f"Number of neurons: {self.number_of_neurons}")
