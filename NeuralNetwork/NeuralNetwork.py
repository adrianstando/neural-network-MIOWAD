import numpy as np
from .Layers import DenseNetLayer
from .Optimizers import Optimizer, SGD


class Net:
    def __init__(self, optimizer: Optimizer = SGD()):
        self.layers = []
        self.optimizer = optimizer

    def add_layer(self, layer: DenseNetLayer):
        self.layers.append(layer)

    def forward(self, vector_in):
        x = vector_in
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x

    def backward_step(self, vector_in, y_true):
        # weighted_inputs <- a
        weighted_inputs = []
        inp = vector_in
        for layer in self.layers:
            w = layer.weighted_input(inp)
            weighted_inputs.append(w)
            inp = layer.activation_function(w)
        # error <- e
        error = []
        number_of_layers = len(self.layers)
        # last layer
        error.append(
            np.multiply(
                self.layers[number_of_layers - 1].activation_function_derivative(
                    weighted_inputs[number_of_layers - 1]
                ),
                # y_hat
                self.layers[number_of_layers - 1].activation_function(
                    weighted_inputs[number_of_layers - 1]
                ) - y_true
            )
        )
        # next layers
        for i in reversed(range(number_of_layers - 1)):
            error.append(
                np.multiply(
                    self.layers[i].activation_function_derivative(
                        weighted_inputs[i]
                    ),
                    error[-1] @ np.transpose(self.layers[i + 1].weights)
                )
            )

        error.reverse()

        # calculate weight changes
        weight_changes = []
        inp = vector_in
        for i in range(len(self.layers)):
            weight_changes.append(
                np.outer(self.layers[i].activation_function(inp), error[i])
            )
            inp = weighted_inputs[i]

        # biases
        biases_changes = error

        return weight_changes, biases_changes

    def train(self, X, Y, X_eval, Y_eval, n_epochs=100, eval_frequency=1):
        for i in range(n_epochs):
            self.optimizer.step(X, Y,
                                [layer.weights for layer in self.layers],
                                [layer.biases for layer in self.layers],
                                self.backward_step)
            if i % eval_frequency == 0:
                print(f"Epoch: {i}, mse train: {mse(Y, self.forward(X))}, mse eval: {mse(Y_eval, self.forward(X_eval))}")
        print(f"Training result:"
              f"    mse train: {mse(Y, self.forward(X))} "
              f"    mse eval: {mse(Y_eval, self.forward(X_eval))}")

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


def mse(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / y_pred.size
