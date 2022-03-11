import numpy as np


class Optimizer:
    def step(self, X, Y, weights, biases, backward_func):
        raise NotImplemented


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads, lr):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * lr * weight_grads[i])
                new_biases.append((-1) * lr * biases_grads[i])

            return new_weights, new_biases

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            change_weights, change_biases = changes(
                weight_grads=backward[0],
                biases_grads=backward[1],
                lr=self.learning_rate
            )
            for j in range(len(weights)):
                weights[j] += change_weights[j]
                biases[j] += change_biases[j]


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * self.learning_rate * weight_grads[i])
                new_biases.append((-1) * self.learning_rate * biases_grads[i])

            return new_weights, new_biases

        weights_changes = [np.zeros_like(x) for x in weights]
        biases_changes = [np.zeros_like(x) for x in biases]

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            single_weight_change, single_bias_change = changes(
                weight_grads=backward[0],
                biases_grads=backward[1]
            )
            for j in range(len(weights_changes)):
                weights_changes[j] = weights_changes[j] + single_weight_change[j]
                biases_changes[j] = biases_changes[j] + single_bias_change[j]

        for j in range(len(weights)):
            weights[j] += weights_changes[j]
            biases[j] += biases_changes[j]


class GradientDescentBatch(Optimizer):
    def __init__(self, learning_rate=1e-3, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * self.learning_rate * weight_grads[i])
                new_biases.append((-1) * self.learning_rate * biases_grads[i])

            return new_weights, new_biases

        def batch_gen(x, y, group_size):
            n = x.shape[0]
            for p in range(0, n, group_size):
                yield x[p:min(p + group_size, n)], y[p:min(p + group_size, n)]

        for (X_batch, Y_batch) in batch_gen(X, Y, self.batch_size):
            weights_changes = [np.zeros_like(x) for x in weights]
            biases_changes = [np.zeros_like(x) for x in biases]

            for x, y in zip(X_batch, Y_batch):
                backward = backward_func(x, y)
                single_weight_change, single_bias_change = changes(
                    weight_grads=backward[0],
                    biases_grads=backward[1]
                )
                for j in range(len(weights_changes)):
                    weights_changes[j] = weights_changes[j] + single_weight_change[j]
                    biases_changes[j] = biases_changes[j] + single_bias_change[j]

            for j in range(len(weights)):
                weights[j] += weights_changes[j]
                biases[j] += biases_changes[j]
