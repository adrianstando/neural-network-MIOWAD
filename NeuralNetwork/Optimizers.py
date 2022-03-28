import numpy as np


class Optimizer:
    def step(self, X, Y, weights, biases, backward_func):
        raise NotImplemented


##################
# basic optimizers
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

        weights_changes = [np.zeros_like(x) for x in weights]
        biases_changes = [np.zeros_like(x) for x in biases]

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            single_weight_change, single_bias_change = changes(
                weight_grads=backward[0],
                biases_grads=backward[1],
                lr=self.learning_rate
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
                    biases_grads=backward[1],
                    lr=self.learning_rate
                )
                for j in range(len(weights_changes)):
                    weights_changes[j] = weights_changes[j] + single_weight_change[j]
                    biases_changes[j] = biases_changes[j] + single_bias_change[j]

            for j in range(len(weights)):
                weights[j] += weights_changes[j]
                biases[j] += biases_changes[j]


#####################
# momentum optimizers
class GradientDescent_Momentum(Optimizer):
    def __init__(self, learning_rate=1e-3, momentum=0.99):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.__weights_momentum = None
        self.__biases_momentum = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * weight_grads[i])
                new_biases.append((-1) * biases_grads[i])

            return new_weights, new_biases

        if self.__weights_momentum is None or self.__biases_momentum is None:
            self.__weights_momentum = [np.zeros_like(x) for x in weights]
            self.__biases_momentum = [np.zeros_like(x) for x in biases]

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

        self.__weights_momentum = [self.__weights_momentum[i] * self.momentum + weights_changes[i]
                                   for i in range(len(self.__weights_momentum))]
        self.__biases_momentum = [self.__biases_momentum[i] * self.momentum + biases_changes[i]
                                  for i in range(len(self.__biases_momentum))]

        for j in range(len(weights)):
            weights[j] += self.__weights_momentum[j] * self.learning_rate
            biases[j] += self.__biases_momentum[j] * self.learning_rate


class SGD_Momentum(Optimizer):
    def __init__(self, learning_rate=1e-3, momentum=0.99):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.__weights_momentum = None
        self.__biases_momentum = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * weight_grads[i])
                new_biases.append((-1) * biases_grads[i])

            return new_weights, new_biases

        if self.__weights_momentum is None or self.__biases_momentum is None:
            self.__weights_momentum = [np.zeros_like(x) for x in weights]
            self.__biases_momentum = [np.zeros_like(x) for x in biases]

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            change_weights, change_biases = changes(
                weight_grads=backward[0],
                biases_grads=backward[1]
            )

            self.__weights_momentum = [self.__weights_momentum[i] * self.momentum + change_weights[i]
                                       for i in range(len(self.__weights_momentum))]
            self.__biases_momentum = [self.__biases_momentum[i] * self.momentum + change_biases[i]
                                      for i in range(len(self.__biases_momentum))]

            for j in range(len(weights)):
                weights[j] += self.__weights_momentum[j] * self.learning_rate
                biases[j] += self.__biases_momentum[j] * self.learning_rate


class GradientDescentBatch_Momentum(Optimizer):
    def __init__(self, learning_rate=1e-3, batch_size=32, momentum=0.99):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum

        self.__weights_momentum = None
        self.__biases_momentum = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append((-1) * weight_grads[i])
                new_biases.append((-1) * biases_grads[i])

            return new_weights, new_biases

        def batch_gen(x, y, group_size):
            n = x.shape[0]
            for p in range(0, n, group_size):
                yield x[p:min(p + group_size, n)], y[p:min(p + group_size, n)]

        if self.__weights_momentum is None or self.__biases_momentum is None:
            self.__weights_momentum = [np.zeros_like(x) for x in weights]
            self.__biases_momentum = [np.zeros_like(x) for x in biases]

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

            self.__weights_momentum = [self.__weights_momentum[i] * self.momentum + weights_changes[i]
                                       for i in range(len(self.__weights_momentum))]
            self.__biases_momentum = [self.__biases_momentum[i] * self.momentum + biases_changes[i]
                                      for i in range(len(self.__biases_momentum))]

            for j in range(len(weights)):
                weights[j] += self.__weights_momentum[j] * self.learning_rate
                biases[j] += self.__biases_momentum[j] * self.learning_rate


####################
# RMSProp optimizers
class GradientDescent_RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-3, beta=0.99, eps=1e-08):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

        self.__weights_E = None
        self.__biases_E = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append(weight_grads[i])
                new_biases.append(biases_grads[i])

            return new_weights, new_biases

        if self.__weights_E is None or self.__biases_E is None:
            self.__weights_E = [np.zeros_like(x) for x in weights]
            self.__biases_E = [np.zeros_like(x) for x in biases]

        weights_g = [np.zeros_like(x) for x in weights]
        biases_g = [np.zeros_like(x) for x in biases]

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            single_weight_change, single_bias_change = changes(
                weight_grads=backward[0],
                biases_grads=backward[1]
            )
            for j in range(len(weights_g)):
                weights_g[j] = weights_g[j] + single_weight_change[j]
                biases_g[j] = biases_g[j] + single_bias_change[j]

        self.__weights_E = [self.beta * self.__weights_E[i] + (1 - self.beta) * (weights_g[i] ** 2)
                            for i in range(len(self.__weights_E))]
        self.__biases_E = [self.beta * self.__biases_E[i] + (1 - self.beta) * (biases_g[i] ** 2)
                           for i in range(len(self.__biases_E))]

        for j in range(len(weights)):
            weights[j] -= self.learning_rate * (weights_g[j] / np.sqrt(self.__weights_E[j]))
            biases[j] -= self.learning_rate * (biases_g[j] / np.sqrt(self.__biases_E[j]))


class SGD_RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-3, beta=0.99, eps=1e-08):
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

        self.__weights_E = None
        self.__biases_E = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append(weight_grads[i])
                new_biases.append(biases_grads[i])

            return new_weights, new_biases

        if self.__weights_E is None or self.__biases_E is None:
            self.__weights_E = [np.zeros_like(x) for x in weights]
            self.__biases_E = [np.zeros_like(x) for x in biases]

        weights_g = [np.zeros_like(x) for x in weights]
        biases_g = [np.zeros_like(x) for x in biases]

        for x, y in zip(X, Y):
            backward = backward_func(x, y)
            change_weights, change_biases = changes(
                weight_grads=backward[0],
                biases_grads=backward[1]
            )

            for j in range(len(weights_g)):
                weights_g[j] = weights_g[j] + change_weights[j]
                biases_g[j] = biases_g[j] + change_biases[j]

            self.__weights_E = [self.beta * self.__weights_E[i] + (1 - self.beta) * (weights_g[i] ** 2)
                                for i in range(len(self.__weights_E))]
            self.__biases_E = [self.beta * self.__biases_E[i] + (1 - self.beta) * (biases_g[i] ** 2)
                               for i in range(len(self.__biases_E))]

            for j in range(len(weights)):
                weights[j] -= self.learning_rate * (weights_g[j] / np.sqrt(self.__weights_E[j]))
                biases[j] -= self.learning_rate * (biases_g[j] / np.sqrt(self.__biases_E[j]))


class GradientDescentBatch_RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-3, batch_size=32, beta=0.99, eps=1e-08):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.eps = eps

        self.__weights_E = None
        self.__biases_E = None

    def step(self, X, Y, weights, biases, backward_func):
        def changes(weight_grads, biases_grads):
            if len(weight_grads) != len(weight_grads):
                print("Wrong sizes!")
                return

            new_weights = []
            new_biases = []
            for i in range(len(weight_grads)):
                new_weights.append(weight_grads[i])
                new_biases.append(biases_grads[i])

            return new_weights, new_biases

        def batch_gen(x, y, group_size):
            n = x.shape[0]
            for p in range(0, n, group_size):
                yield x[p:min(p + group_size, n)], y[p:min(p + group_size, n)]

        if self.__weights_E is None or self.__biases_E is None:
            self.__weights_E = [np.zeros_like(x) for x in weights]
            self.__biases_E = [np.zeros_like(x) for x in biases]

        weights_g = [np.zeros_like(x) for x in weights]
        biases_g = [np.zeros_like(x) for x in biases]

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

            self.__weights_E = [self.beta * self.__weights_E[i] + (1 - self.beta) * (weights_g[i] ** 2)
                                for i in range(len(self.__weights_E))]
            self.__biases_E = [self.beta * self.__biases_E[i] + (1 - self.beta) * (biases_g[i] ** 2)
                               for i in range(len(self.__biases_E))]

            for j in range(len(weights)):
                weights[j] -= self.learning_rate * (weights_g[j] / np.sqrt(self.__weights_E[j]))
                biases[j] -= self.learning_rate * (biases_g[j] / np.sqrt(self.__biases_E[j]))
