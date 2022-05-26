import numpy as np
import math
import copy

class UniformGenerator:
    def __init__(self, low, top):
        self.low = low
        self.top = top
        
    def generate(self, dims):
        return np.random.uniform(self.low, self.top, dims)

class GeneticAlgorithm:
    def __init__(self, vector_dim, func):
        self.population = None
        self.vector_dim = vector_dim
        
        self.func = func
        self.__last_population_evaluation = None
    
    def generate_population(self, population_size, generator=UniformGenerator(-1, 1)):
        self.population = [generator.generate(self.vector_dim) for i in range(population_size)]
        self.__last_population_evaluation = [self.func(elem) for elem in self.population]
        
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
            position_to_cut = np.random.randint(1, self.vector_dim)
            
            random_indexes = [
                    indexes_to_crossover.pop(np.random.randint(0, len(indexes_to_crossover))), 
                    indexes_to_crossover.pop(np.random.randint(0, len(indexes_to_crossover)))
            ]
            
            x1 = copy.copy(self.population[random_indexes[0]][0:position_to_cut])
            x2 = copy.copy(self.population[random_indexes[1]][0:position_to_cut])
            
            self.population[random_indexes[0]][0:position_to_cut] = x2
            self.population[random_indexes[1]][0:position_to_cut] = x1
    
    def gauss_mutation(self, mutation_population_ratio=0.2):
        number_of_elements_to_mutate = math.ceil(len(self.population) * mutation_population_ratio)
        indexes_to_mutate = list(np.random.choice(len(self.population), number_of_elements_to_mutate, replace=False))
        for i in range(len(indexes_to_mutate)):
            self.population[indexes_to_mutate[i]] += np.random.normal(size=self.vector_dim)
        
    def selection(self, k=None):
        # selekcja turniejowa
        if k is None:
            k = math.ceil(len(self.population) * 0.3)
            
        new_population=[]
        while(len(new_population) < len(self.population)):
            T = np.random.choice(len(self.population), k, replace=False)
            competition_data = [(T[i], self.__last_population_evaluation[T[i]]) for i in range(len(T))]
            sort = sorted(competition_data, key=lambda tup: tup[1])
            
            # zadaniem jest optymalizacja funkcji (znajdowanie minimum)
            # a zatem wybieram osobnika z najmniejszą wartością funckji
            
            index_to_add = sort[0][0]
            new_population.append(copy.copy(self.population[index_to_add]))
            
        self.population = new_population
        self.__last_population_evaluation = [self.func(elem) for elem in self.population]

    def train(self, n_epochs=100, crossover_population_ratio=0.7, mutation_population_ratio=0.2, k=None, eval_frequency=10):
        for i in range(1, n_epochs + 1):
            self.crossover(crossover_population_ratio)
            self.gauss_mutation(mutation_population_ratio)
            self.selection(k)
            
            if i % eval_frequency == 0:
                min_value = min(self.__last_population_evaluation)
                print(f"Epoch: {i}/{n_epochs}")
                print(f"Best solution function value: {min_value}")
        
        if i % eval_frequency != 0:
            min_value = min(self.__last_population_evaluation)
            print(f"Epoch: {i}/{n_epochs}")
            print(f"Best solution function value: {min_value}")
                
        
        
