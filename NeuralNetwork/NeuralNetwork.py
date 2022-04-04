import numpy as np
import matplotlib.pyplot as plt
from .Layers import DenseNetLayer
from .Optimizers import Optimizer, SGD


class Net:
    def __init__(self, optimizer: Optimizer = SGD(), loss: str = 'mse'):
        self.layers = []
        self.optimizer = optimizer
        if loss == 'mse':
            self.loss_name = 'mse'
            self.loss_func = mse
        elif loss == 'cross-entropy':
            self.loss_name = 'cross-entropy'
            self.loss_func = cross_entropy
        else:
            raise Exception('Unknown loss function')

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
                print(
                    f"Epoch: {i}, {self.loss_name} train: {self.loss_func(Y, self.forward(X))}, {self.loss_name} eval: {self.loss_func(Y_eval, self.forward(X_eval))}")
        print(f"Training result:")
        print(f"    {self.loss_name} train: {self.loss_func(Y, self.forward(X))}")
        print(f"    {self.loss_name} eval: {self.loss_func(Y_eval, self.forward(X_eval))}")

    def train_and_visualize(self, X, Y, X_eval, Y_eval, n_epochs=100, eval_frequency=1):
        weight_norms = []
        bias_norms = []
        mse_train = []
        mse_test = []

        def __change_to_2d(x):
            if x.ndim == 1:
                return np.reshape(x, (-1, 1))
            else:
                return x

        for i in range(n_epochs):
            self.optimizer.step(X, Y,
                                [layer.weights for layer in self.layers],
                                [layer.biases for layer in self.layers],
                                self.backward_step)
            if i % eval_frequency == 0:
                mse_single_train = self.loss_func(Y, self.forward(X))
                mse_single_test = self.loss_func(Y_eval, self.forward(X_eval))
                mse_train.append(mse_single_train)
                mse_test.append(mse_single_test)
                # print(f"Epoch: {i}, mse train: {mse_single_train}, mse eval: {mse_single_test}")
            weight_norms.append([np.linalg.norm(__change_to_2d(layer.weights), ord='fro') for layer in self.layers])
            bias_norms.append([np.linalg.norm(__change_to_2d(layer.biases), ord='fro') for layer in self.layers])

        print(f"Training result:")
        print(f"    {self.loss_name} train: {self.loss_func(Y, self.forward(X))}")
        print(f"    {self.loss_name} eval: {self.loss_func(Y_eval, self.forward(X_eval))}")

        for i in range(len(self.layers)):
            plt.scatter(
                list(range(n_epochs)),
                [weight_norms[j][i] for j in range(n_epochs)]
            )
            plt.title(f"Norm of weights of layer number: {i}")
            plt.show()

        for i in range(len(self.layers)):
            plt.scatter(
                list(range(n_epochs)),
                [bias_norms[j][i] for j in range(n_epochs)]
            )
            plt.title(f"Norm of biases of layer number: {i}")
            plt.show()

        plt.scatter(
            list(range(0, n_epochs, eval_frequency)),
            mse_train,
            color='blue'
        )
        plt.scatter(
            list(range(0, n_epochs, eval_frequency)),
            mse_test,
            color='red'
        )
        plt.title(f"{self.loss_name} of train and test set")
        plt.legend(['train', 'test'])
        plt.show()

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


def f1_score(y_true, y_pred):
    # input has shape:
    #                   class0      class1
    # observation1      ...         ...
    # observation2      ...         ...
    # observation3      ...         ...
    # ...
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(y_true)):
        if y_true[i, 1] == 1 and y_pred[i, 1] == 1:
            TP += 1
        elif y_true[i, 0] == 1 and y_pred[i, 0] == 1:
            TN += 1
        elif y_true[i, 0] == 1 and y_pred[i, 1] == 1:
            FP += 1
        else:
            FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return 2 * (precision * recall) / (precision + recall)


def f1_score_macro(y_true, y_pred):
    # input has shape:
    #                   class0      class1      class2      ...
    # observation1      ...         ...         ...         ...
    # observation2      ...         ...         ...         ...
    # observation3      ...         ...         ...         ...
    # ...
    
    def f1_score_one_dim(y_true, y_pred):
        TP = 0
        FN = 0
        FP = 0
        TN = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            else:
                FN += 1
        
        #print(TP, FN, FP, TN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        return 2 * (precision * recall) / (precision + recall)
    
    n_class = y_true.shape[1]
    outs = []
    for i in range(n_class):
        outs.append(
            f1_score_one_dim(y_true[:, i], y_pred[:, i])
        )

    return np.mean(outs)


def cross_entropy(y_true, y_pred):   
    # input has shape:
    #                   class0      class1      class2      ...
    # observation1      ...         ...         ...         ...
    # observation2      ...         ...         ...         ...
    # observation3      ...         ...         ...         ...
    # ...
    
    epsilon = 1e-12
    predictions = np.clip(np.array(y_pred), epsilon, 1. - epsilon)

    n_class = y_true.shape[1]
    n = y_true.shape[0]

    out = 0

    for i in range(n):
        for j in range(n_class):
            out -= y_true[i, j] * np.log(predictions[i, j])

    return out / n
