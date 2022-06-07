import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
import time
import random
import math


class Rectangle:
    def __init__(self, height, width, val):
        self.height = height
        self.width = width
        self.val = val
    
    def __repr__(self):
        return f"Rectangle: width={self.width}, height={self.height}, value={self.val}"


class Orientation(Enum):
    HORIZONTAL = 'h'
    VERTICAL = 'v'
    
class Stripe:
    def __init__(self, rectangles, rectangle_orientation: Orientation = Orientation.HORIZONTAL, stack_orient: Orientation = Orientation.HORIZONTAL):
        self.stripe_height = 0
        self.stripe_width = 0
        
        if rectangle_orientation == 'h' and stack_orient == 'h':
            assert all(x.height == rectangles[0].height for x in rectangles), "Elements do not have the same height!"
            self.stripe_height = rectangles[0].height
            self.stripe_width = np.sum([r.width for r in rectangles])
        elif rectangle_orientation == 'h' and stack_orient == 'v':
            assert all(x.width == rectangles[0].width for x in rectangles), "Elements do not have the same width!"
            self.stripe_height = np.sum([r.height for r in rectangles])
            self.stripe_width = rectangles[0].width
        elif rectangle_orientation == 'v' and stack_orient == 'h':
            assert all(x.height == rectangles[0].height for x in rectangles), "Elements do not have the same width!"
            self.stripe_height = np.sum([r.width for r in rectangles])
            self.stripe_width = rectangles[0].height
        elif rectangle_orientation == 'v' and stack_orient == 'v':
            assert all(x.width == rectangles[0].width for x in rectangles), "Elements do not have the same width!"
            self.stripe_height = rectangles[0].width
            self.stripe_width = np.sum([r.height for r in rectangles])
        else:
            raise Exception('Wrong directions!')
            
        self.rectangle_orientation = rectangle_orientation
        self.stack_orient = stack_orient
        self.rectangles = rectangles
        self.stripe_val = np.sum([r.val for r in rectangles])
    
    def __repr__(self):
        return f"Stripe: height={self.stripe_height}, width={self.stripe_width}, value={self.stripe_val}"
    

class Circle:
    def __init__(self, r):
        self.r = r
        self.stripes = []
    
    def val_evaluate(self):
        return np.sum([stripe.stripe.stripe_val for stripe in self.stripes])
        
    class StripePosition:
        def __init__(self, stripe, position):
            self.stripe = stripe
            
            self.left_down = position
            self.left_up = (position[0], position[1] + stripe.stripe_height)
            self.right_down = (position[0] + stripe.stripe_width, position[1])
            self.right_up = (position[0] + stripe.stripe_width, position[1] + stripe.stripe_height)
        
        def overlap(self, other):
            # left side between vertical sides
            if self.left_up[0] <= other.left_up[0] < self.right_up[0]:
                if other.left_up[1] > self.left_up[1] and other.left_down[1] < self.left_up[1]:
                    return True
                if other.left_up[1] <= self.left_up[1] and other.left_down[1] >= self.left_down[1]:
                    return True
                if other.left_down[1] < self.left_down[1] and other.left_up[1] > self.left_down[1]:
                    return True
                if other.left_up[1] >= self.left_up[1] and other.left_down[1] <= self.left_down[1]:
                    return True
                
            # right side between vertical sides
            if self.left_up[0] < other.right_up[0] <= self.right_up[0]:
                if other.left_up[1] > self.left_up[1] and other.left_down[1] < self.left_up[1]:
                    return True
                if other.left_up[1] <= self.left_up[1] and other.left_down[1] >= self.left_down[1]:
                    return True
                if other.left_down[1] < self.left_down[1] and other.left_up[1] > self.left_down[1]:
                    return True
                if other.left_up[1] >= self.left_up[1] and other.left_down[1] <= self.left_down[1]:
                    return True
            
            # so now neither right side nor left side is between verical sides of self  
            if other.left_up[0] < self.left_up[0] and other.right_up[0] > self.right_up[0]:
                if other.left_up[1] > self.left_up[1] and other.left_down[1] < self.left_up[1]:
                    return True
                if other.left_up[1] <= self.left_up[1] and other.left_down[1] >= self.left_down[1]:
                    return True
                if other.left_up[1] > self.left_down[1] and other.left_down[1] < self.left_down[1]:
                    return True
                if other.left_up[1] >= self.left_up[1] and other.left_down[1] <= self.left_down[1]:
                    return True
                
            return False
        
        def __repr__(self):
            return f"Stripe position: left_down={self.left_down}, right_up={self.right_up}"
 
    def add_stripe(self, x, y, stripe, force=False):
        new = Circle.StripePosition(stripe, (x, y))
        
        # circle condition
        # let the center of the circle be (0,0)
        points = [new.left_down, new.left_up, new.right_down, new.right_up]
        distances = [(p[0] ** 2 + p[1] ** 2) ** (1/2) for p in points]
        fits_inside = np.all(np.array(distances) <= self.r)
        
        if not fits_inside:
            return False
        
        if force:
            self.stripes = [s for s in self.stripes if not new.overlap(s)]
            self.stripes.append(new)
            return True
        else:
            overlaping = [s for s in self.stripes if new.overlap(s)]
            
            if len(overlaping) == 0:
                self.stripes.append(new)
                return True
            else:
                return False
            
    def move_stripe(self, x_change, y_change, stripe_ind, force=False):
        old = self.stripes[stripe_ind]
        new = Circle.StripePosition(old.stripe, (old.left_down[0] + x_change, old.left_down[1] + y_change))
        
        # circle condition
        # let the center of the circle be (0,0)
        points = [new.left_down, new.left_up, new.right_down, new.right_up]
        distances = [(p[0] ** 2 + p[1] ** 2) ** (1/2) for p in points]
        fits_inside = np.all(np.array(distances) <= self.r)
        
        if not fits_inside:
            return False
        
        if force:
            self.stripes = [s for s in self.stripes if not new.overlap(s)]
            self.stripes.append(new)
            return True
        else:
            circle_copy = copy.deepcopy(self)
            circle_copy.stripes.pop(stripe_ind)
            if len([s for s in circle_copy.stripes if new.overlap(s)]) == 0:
                self.stripes[stripe_ind] = new
                return True
            else:
                return False
            
    def plot(self, figsize=(8, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        cir = plt.Circle((0, 0), self.r, color='black', fill=False, linewidth=2)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
        
        for s in self.stripes:
            height = s.stripe.stripe_height
            width = s.stripe.stripe_width
            rect = patches.Rectangle(s.left_down, width, height, facecolor='r', edgecolor='black', linewidth=3)
            ax.add_patch(rect)
            
            if s.stripe.rectangle_orientation == 'h' and s.stripe.stack_orient == 'h':
                added_param = 0                
                for rect in s.stripe.rectangles:
                    p = np.array(copy.copy(s.left_down))
                    p[0] += added_param
                    
                    rect_plot = patches.Rectangle(p, rect.width, rect.height, facecolor='none', edgecolor='black', linewidth=1)
                    ax.add_patch(rect_plot)
                    added_param += rect.width
            elif s.stripe.rectangle_orientation == 'h' and s.stripe.stack_orient == 'v':
                added_param = 0                
                for rect in s.stripe.rectangles:
                    p = np.array(copy.copy(s.left_down))
                    p[1] += added_param
                    
                    rect_plot = patches.Rectangle(p, rect.width, rect.height, facecolor='none', edgecolor='black', linewidth=1)
                    ax.add_patch(rect_plot)
                    added_param += rect.height
            elif s.stripe.rectangle_orientation == 'v' and s.stripe.stack_orient == 'h':
                added_param = 0                
                for rect in s.stripe.rectangles:
                    p = np.array(copy.copy(s.left_down))
                    p[1] += added_param
                    
                    rect_plot = patches.Rectangle(p, rect.height, rect.width, facecolor='none', edgecolor='black', linewidth=1)
                    ax.add_patch(rect_plot)
                    added_param += rect.width
            elif s.stripe.rectangle_orientation == 'v' and s.stripe.stack_orient == 'v':
                added_param = 0                
                for rect in s.stripe.rectangles:
                    p = np.array(copy.copy(s.left_down))
                    p[0] += added_param
                    
                    rect_plot = patches.Rectangle(p, rect.height, rect.width, facecolor='none', edgecolor='black', linewidth=1)
                    ax.add_patch(rect_plot)
                    added_param += rect.height
        
        plt.xlim((-self.r * 1.1, self.r * 1.1))
        plt.ylim((-self.r * 1.1, self.r * 1.1))
        plt.axis('off')
        
    def __random_point_from_circle(self):   
        circle_r = self.r
        alpha = 2 * math.pi * random.random()
        r = circle_r * math.sqrt(random.random())
        
        x = r * math.cos(alpha)
        y = r * math.sin(alpha)
        
        return(x, y)
    
    def fill_with_stripes(self, stripes, rectangles_limit=50, time_limit=3):
        # these parameters are introduced to make sure that the function finishes 
        # calculations in reasonable time:
        #
        # rectangles_limit - maximum number of rectangles, which will be added
        # time_limit - maximum time in seconds for filling the circle
        
        i = 0
        start_time = time.time()
        
        while i < rectangles_limit and time.time() - start_time < time_limit:
            p = self.__random_point_from_circle()
            choice = random.choice(stripes)
            result = self.add_stripe(p[0], p[1], choice)
            if result is True: 
                i += 1


def stripe_generator(rectangles, max_width):
    stripe_list = []
    
    def insert_new_and_add_to__stripe_list(current_solution, rectangles_possible, max_size, stripe_list, rectangle_orientation, stack_orient):
        current_size = 0
        
        if len(current_solution) != 0:
            if rectangle_orientation == 'h' and stack_orient == 'h':
                current_size = np.sum([r.width for r in current_solution])
            elif rectangle_orientation == 'h' and stack_orient == 'v':
                current_size = np.sum([r.height for r in current_solution])
            elif rectangle_orientation == 'v' and stack_orient == 'h':
                current_size = np.sum([r.width for r in current_solution])
            elif rectangle_orientation == 'v' and stack_orient == 'v':
                current_size = np.sum([r.height for r in current_solution])
            else:
                return
                
            if current_size <= max_size:
                sol = copy.deepcopy(current_solution)
                stripe_list.append(Stripe(sol, rectangle_orientation, stack_orient))
            else: 
                return
        
        for r in rectangles_possible:
            sol = copy.deepcopy(current_solution)
            sol.append(r)
            insert_new_and_add_to__stripe_list(sol, rectangles_possible, max_size, stripe_list, rectangle_orientation, stack_orient)
           
    height_list = list(set([r.height for r in rectangles]))
    width_list = list(set([r.width for r in rectangles]))
    
    for h in height_list:
        possible_rectangles = [r for r in rectangles if r.height == h]
        solution = []
        insert_new_and_add_to__stripe_list(solution, possible_rectangles, max_width, stripe_list, 'h', 'h')
    for w in width_list:
        possible_rectangles = [r for r in rectangles if r.width == w]
        solution = []
        insert_new_and_add_to__stripe_list(solution, possible_rectangles, max_width, stripe_list, 'h', 'v')
    for h in height_list:
        possible_rectangles = [r for r in rectangles if r.height == h]
        solution = []
        insert_new_and_add_to__stripe_list(solution, possible_rectangles, max_width, stripe_list, 'v', 'h')
    for w in width_list:
        possible_rectangles = [r for r in rectangles if r.width == w]
        solution = []
        insert_new_and_add_to__stripe_list(solution, possible_rectangles, max_width, stripe_list, 'v', 'v')
    
    return stripe_list


def simple_stripe_generator(rectangles):
    # stripes with only one rectangle
    stripe_list = []      
        
    for rect in rectangles:
        stripe_list.append(Stripe([rect], 'h', 'h'))
        stripe_list.append(Stripe([rect], 'h', 'v'))
        stripe_list.append(Stripe([rect], 'v', 'h'))
        stripe_list.append(Stripe([rect], 'v', 'v'))
    
    return stripe_list
    
        
def rectangles_from_file(path):
    out_list = []
    
    with open(path) as fp:
        lines = fp.readlines()
        for index, line in enumerate(lines):
            lst = line.split(',')
            out_list.append(Rectangle(float(lst[1]), float(lst[0]), float(lst[2])))
            
    return out_list


class CirclePopulation:
    def __init__(self, r, stripes, n=100):
        self.stripes = stripes
        self.n = n
        self.r = r
        self.population = [None for i in range(n)]
      
    def generate(self, *args, **kwargs):
        for i in range(self.n):
            self.population[i] = Circle(self.r)
            self.population[i].fill_with_stripes(self.stripes, *args, **kwargs)
    
    def crossover(self, crossover_population_ratio=0.7, stripe_ratio_to_cut=None):
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
             
             # co jak zero stripe'ów?
             n_elements_1_to_cut, n_elements_2_to_cut = None, None
             if stripe_ratio_to_cut is None:
                 n_elements_1_to_cut = np.random.randint(1, len(self.population[random_indexes[0]].stripes))
                 n_elements_2_to_cut = np.random.randint(1, len(self.population[random_indexes[1]].stripes))
             else:
                 n_elements_1_to_cut = math.ceil(len(self.population[random_indexes[0]].stripes) * stripe_ratio_to_cut)
                 if n_elements_1_to_cut > len(self.population[random_indexes[0]].stripes):
                     n_elements_1_to_cut = len(self.population[random_indexes[0]].stripes)
                
                 n_elements_2_to_cut = math.ceil(len(self.population[random_indexes[1]].stripes) * stripe_ratio_to_cut)
                 if n_elements_2_to_cut > len(self.population[random_indexes[1]].stripes):
                     n_elements_2_to_cut = len(self.population[random_indexes[1]].stripes)
             
             # wyciągnij losowo stripy i wsadz do drugiego
             circle_1_copy = copy.deepcopy(self.population[random_indexes[0]])
             circle_2_copy = copy.deepcopy(self.population[random_indexes[1]])
             
             for i in range(n_elements_1_to_cut):
                 stripe = circle_1_copy.stripes.pop(np.random.randint(0, len(circle_1_copy.stripes)))
                 self.population[random_indexes[1]].add_stripe(stripe.left_down[0], stripe.left_down[1], stripe.stripe, force=True)
            
             for i in range(n_elements_2_to_cut):
                 stripe = circle_2_copy.stripes.pop(np.random.randint(0, len(circle_2_copy.stripes)))
                 self.population[random_indexes[0]].add_stripe(stripe.left_down[0], stripe.left_down[1], stripe.stripe, force=True)
                 
                 
    
    def mutation(self, mutation_population_ratio=0.2, rectangles_limit=5, time_limit=1/10):
        number_of_elements_to_mutate = math.ceil(len(self.population) * mutation_population_ratio)
        indexes_to_mutate = list(np.random.choice(len(self.population), number_of_elements_to_mutate, replace=False))
        
        for i in range(len(indexes_to_mutate)):
            for j in range(len(self.population[indexes_to_mutate[i]].stripes)):
                self.population[indexes_to_mutate[i]].move_stripe(np.random.normal(), np.random.normal(), j)
            
            self.population[indexes_to_mutate[i]].fill_with_stripes(self.stripes, rectangles_limit=rectangles_limit, time_limit=time_limit)
    
    def selection(self, k=None):
        # k - ile osobników do turnieju
        
        # selekcja turniejowa
        if k is None:
            k = math.ceil(len(self.population) * 0.3)
            
        new_population=[]
        while(len(new_population) < len(self.population)):
            T = np.random.choice(len(self.population), k, replace=False)
            competition_data = [(T[i], self.population[T[i]].val_evaluate()) for i in range(len(T))]
            sort = sorted(competition_data, key=lambda tup: tup[1], reverse=True)
            
            # zadaniem jest znalezienie osobnika z jak najwyższą funkcją przystosowania
            # a zatem wybieram osobnika z najwięszą wartością funckji
            
            index_to_add = sort[0][0]
            new_population.append(copy.copy(self.population[index_to_add]))
        
        self.population = new_population
    
    def train(self, n_epochs=100, 
              crossover_population_ratio=0.7, stripe_ratio_to_cut=None, 
              mutation_population_ratio=0.2, rectangles_limit=5, time_limit=1/10,
              k=None, 
              eval_frequency=10, eval_score_finish=None, history=False):
        
        eval_tab = []
        eval_tab.append(max([p.val_evaluate() for p in self.population]))
        
        for i in range(1, n_epochs + 1):
            self.crossover(crossover_population_ratio, stripe_ratio_to_cut)
            self.mutation(mutation_population_ratio, rectangles_limit, time_limit)
            self.selection(k)
            
            if i % eval_frequency == 0:
                max_value = max([p.val_evaluate() for p in self.population])
                print(f"Epoch: {i}/{n_epochs}")
                print(f"Best solution function value: {max_value}")
                
                if history:
                    eval_tab.append(max_value)
                
                if eval_score_finish is not None:
                    if max_value > eval_score_finish:
                        break
        
        if i % eval_frequency != 0:
            max_value = max([p.val_evaluate() for p in self.population])
            print(f"Epoch: {i}/{n_epochs}")
            print(f"Best solution function value: {max_value}")
            
        if history:
            return eval_tab
    
    
    
    