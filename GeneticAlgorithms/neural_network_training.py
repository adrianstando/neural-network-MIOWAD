import numpy as np
import math
import copy

from NeuralNetwork.NeuralNetwork import Net, mse
from NeuralNetwork.ActivationFunctions import SigmoidFunction, LinearFunction
from NeuralNetwork.Initializers import RandomNormalInitializer
from NeuralNetwork.Layers import DenseNetLayer


def simple_network_generator():
    net = Net()
    net.add_layer(DenseNetLayer(1, 20, SigmoidFunction(), RandomNormalInitializer()))
    net.add_layer(DenseNetLayer(20, 5, SigmoidFunction(), RandomNormalInitializer()))
    net.add_layer(DenseNetLayer(5, 1, LinearFunction(), RandomNormalInitializer()))
    return net


class NeuralNetworkPopulation:
     def __init__(self, n=100, func=mse):
         self.n = n
         self.func = func
         self.population = []
         self.population_evaluation = []
    
     def generate_population(self, network_generating_func=simple_network_generator):
         self.population = [network_generating_func() for i in range(self.n)]
         
     def evaluate(self, X, Y):     
         if (len(self.population_evaluation) == 0):
            return min([self.func(Y, self.population[i].forward(X)) for i in range(len(self.population))])
         else:
            return min(self.population_evaluation)
     
     def crossover(self, crossover_population_ratio=0.7):
         number_of_elements_to_crossover = math.ceil(len(self.population) * crossover_population_ratio)
         if number_of_elements_to_crossover % 2 == 1:
             if number_of_elements_to_crossover + 1 <= len(self.population):
                 number_of_elements_to_crossover += 1
             elif number_of_elements_to_crossover - 1 >= 0:
                 number_of_elements_to_crossover -= 1
             else:
                 return
        
         indexes_to_crossover = list(np.random.choice(len(self.population), number_of_elements_to_crossover, replace=False))
         
         for i in range(number_of_elements_to_crossover // 2):
             random_indexes = [
                    indexes_to_crossover.pop(np.random.randint(0, len(indexes_to_crossover))), 
                    indexes_to_crossover.pop(np.random.randint(0, len(indexes_to_crossover)))
             ]
             
             # biases
             for i in range(len(self.population[random_indexes[0]].layers)):
                 if len(self.population[random_indexes[0]].layers[i].biases) == 1:
                     x1 = copy.deepcopy(self.population[random_indexes[0]].layers[i].biases)
                     x2 = copy.deepcopy(self.population[random_indexes[1]].layers[i].biases)
                     self.population[random_indexes[0]].layers[i].biases = x2
                     self.population[random_indexes[1]].layers[i].biases = x1
                 else:
                     position_to_cut = np.random.randint(1, len(self.population[random_indexes[0]].layers[i].biases))
                     x1 = copy.deepcopy(self.population[random_indexes[0]].layers[i].biases[0:position_to_cut])
                     x2 = copy.deepcopy(self.population[random_indexes[1]].layers[i].biases[0:position_to_cut])
                     self.population[random_indexes[0]].layers[i].biases[0:position_to_cut] = x2
                     self.population[random_indexes[1]].layers[i].biases[0:position_to_cut] = x1
             
             # weights
             for i in range(len(self.population[random_indexes[0]].layers)):
                 if self.population[random_indexes[0]].layers[i].weights.shape[0] == 1:
                     position_to_cut_0 = np.random.randint(1, self.population[random_indexes[0]].layers[i].weights.shape[1])
                     
                     x1 = copy.deepcopy(self.population[random_indexes[0]].layers[i].weights[0, 0:position_to_cut_0])
                     x2 = copy.deepcopy(self.population[random_indexes[1]].layers[i].weights[0, 0:position_to_cut_0])
                     self.population[random_indexes[0]].layers[i].weights[0, 0:position_to_cut_0] = x2
                     self.population[random_indexes[1]].layers[i].weights[0, 0:position_to_cut_0] = x1
                 elif self.population[random_indexes[0]].layers[i].weights.shape[1] == 1:
                     position_to_cut_0 = np.random.randint(1, self.population[random_indexes[0]].layers[i].weights.shape[0])
                     
                     x1 = copy.deepcopy(self.population[random_indexes[0]].layers[i].weights[0:position_to_cut_0, 0])
                     x2 = copy.deepcopy(self.population[random_indexes[1]].layers[i].weights[0:position_to_cut_0, 0])
                     self.population[random_indexes[0]].layers[i].weights[0:position_to_cut_0, 0] = x2
                     self.population[random_indexes[1]].layers[i].weights[0:position_to_cut_0, 0] = x1
                 else:
                     position_to_cut_0 = np.random.randint(1, self.population[random_indexes[0]].layers[i].weights.shape[0])
                     position_to_cut_1 = np.random.randint(1, self.population[random_indexes[0]].layers[i].weights.shape[1])
                     
                     x1 = copy.deepcopy(self.population[random_indexes[0]].layers[i].weights[0:position_to_cut_0, 0:position_to_cut_1])
                     x2 = copy.deepcopy(self.population[random_indexes[1]].layers[i].weights[0:position_to_cut_0, 0:position_to_cut_1])
                     self.population[random_indexes[0]].layers[i].weights[0:position_to_cut_0, 0:position_to_cut_1] = x2
                     self.population[random_indexes[1]].layers[i].weights[0:position_to_cut_0, 0:position_to_cut_1] = x1
     
     def mutation(self, mutation_population_ratio=0.2, mu=0, sigma=10):
         number_of_elements_to_mutate = math.ceil(len(self.population) * mutation_population_ratio)
         indexes_to_mutate = list(np.random.choice(len(self.population), number_of_elements_to_mutate, replace=False))
         
         for i in range(number_of_elements_to_mutate) :
             for layer in self.population[indexes_to_mutate[i]].layers:
                 layer.weights += np.random.normal(mu, sigma, size = layer.weights.shape)
                 layer.biases += np.random.normal(mu, sigma, size = layer.biases.shape)
     
     def selection(self, X, Y, k=None):
         # selekcja turniejowa
         if k is None:
             k = math.ceil(len(self.population) * 0.3)
         
         if (len(self.population_evaluation) == 0):
             self.population_evaluation = [self.func(Y, self.population[i].forward(X)) for i in range(len(self.population))]
        
         new_population=[]
         while(len(new_population) < len(self.population)):
             T = np.random.choice(len(self.population), k, replace=False)
             competition_data = [(T[i], self.population_evaluation[T[i]]) for i in range(len(T))]
             sort = sorted(competition_data, key=lambda tup: tup[1])
        
             # zadaniem jest optymalizacja funkcji (znajdowanie minimum)
             # a zatem wybieram osobnika z najmniejszą wartością funckji
            
             index_to_add = sort[0][0]
             new_population.append(copy.deepcopy(self.population[index_to_add]))
        
         self.population = new_population
         self.population_evaluation = [self.func(Y, self.population[i].forward(X)) for i in range(len(self.population))]
     
     def train(self, X, Y, n_epochs=50, eval_frequency=10, crossover_population_ratio=0.7, mutation_population_ratio=0.2, mutation_mu=0, mutation_sigma=1, mutation_sigma_learning_rate=0.9, k=None, return_history=False):
         sigma = mutation_sigma
         
         history = []
         
         for i in range(1, n_epochs + 1):
            self.crossover(crossover_population_ratio)
            self.mutation(mutation_population_ratio, mutation_mu, sigma)
            self.selection(X, Y, k)
            
            if i % eval_frequency == 0:
                min_value = min(self.population_evaluation)
                history.append(min_value)
                print(f"Epoch: {i}/{n_epochs}")
                print(f"Best solution function value: {min_value}")
            
            sigma = sigma * mutation_sigma_learning_rate
        
         if i % eval_frequency != 0:
            min_value = min(self.population_evaluation)
            history.append(min_value)
            print(f"Epoch: {i}/{n_epochs}")
            print(f"Best solution function value: {min_value}")
            
         if return_history:
            return history
