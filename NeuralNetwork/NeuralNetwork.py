import numpy as np


class DenseNetLayer:
    def __init__(self, input_size, number_of_neurons, activation_function):
        self.input_size = input_size
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.rand(number_of_neurons, input_size)
        self.biases = np.random.rand(number_of_neurons)
        self.activation_function = self.__generate_vectorized_function(activation_function)

    @staticmethod
    def __generate_vectorized_function(func):
        f = np.vectorize(func)
        return lambda x: f(x)

    def out(self, vector_in):
        y = vector_in @ self.weights
        y = y + self.biases
        return self.activation_function(y)

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


class Net:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def out(self, vector_in):
        x = vector_in
        for i in range(len(self.layers)):
            x = self.layers[i].out(x)
        return x

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

    def set_biases(self, biases):
        for i in range(len(self.layers)):
            self.layers[i].set_biases(biases[i])

    def summary(self):
        print("Model summary")
        print(" ")
        for i in range(len(self.layers)):
            print(f"Layer {i}")
            self.layers[i].summary()
            print(" ")


def sigmoid_function(x):
    return np.exp(x) / (np.exp(x) + 1)


def linear_function(x):
    return x


def mse(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / y_pred.size


if __name__ == '__main__':
    v_in = np.array([2, 1])

    l1 = DenseNetLayer(input_size=2, number_of_neurons=2, activation_function=linear_function)
    l1.set_weights(np.array([[0.45, 0.32], [-0.12, 0.29]]))
    l1.set_biases(np.array([0, 0]))

    l2 = DenseNetLayer(2, 2, linear_function)
    l2.set_weights(np.array([[0.48, -0.12], [0.64, 0.91]]))
    l2.set_biases(np.array([0, 0]))
    print("Layer by layer example:")
    print(l2.out(l1.out(v_in)))

    n = Net()
    n.add_layer(DenseNetLayer(2, 2, linear_function))
    n.add_layer(DenseNetLayer(2, 2, linear_function))
    n.layers[0].set_weights(np.array([[0.45, 0.32], [-0.12, 0.29]]))
    n.layers[0].set_biases(np.array([0, 0]))
    n.layers[1].set_weights(np.array([[0.48, -0.12], [0.64, 0.91]]))
    n.layers[1].set_biases(np.array([0, 0]))
    print("Full net example:")
    print(n.out(v_in))

    v_in = np.array([[2, 1], [2, 1]])
    print("Two rows example:")
    print(n.out(v_in))

    v_in = np.array([[2, 1], [2, 1], [2, 1], [2, 1], [2, 1]])
    print("Many rows example:")
    print(n.out(v_in))

    print("L1 output example with one and two rows:")
    v_in = np.array([2, 1])
    print(l1.out(v_in))
    v_in = np.array([[2, 1], [2, 1]])
    print(l1.out(v_in))

    print(" ")
    n.summary()
