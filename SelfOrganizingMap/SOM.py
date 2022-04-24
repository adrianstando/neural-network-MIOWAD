import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

class SOM:
    def __init__(self, width, height, vector_in_size, neighbourhood_function, learning_rate_function, weight_initializer, neighbourhood_scaler=1):
        self.width = width
        self.height = height
        self.vector_in_size = vector_in_size
        self.neighbourhood_function = neighbourhood_function
        self.learning_rate_function = learning_rate_function
        self.neighbourhood_scaler = neighbourhood_scaler
        
        self.random_generator = np.random.default_rng()
        
        self.weights = weight_initializer.initialize(width, height, vector_in_size)
        self.n_epochs = 0
    
    def find_BMU(self, x):
        #https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
        distances = (np.square(self.weights - x[..., np.newaxis, np.newaxis])).sum(axis=0) * self.neighbourhood_scaler
        return np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    
    def __update_weights(self, BMU, obs, epoch):
        for w in range(self.width):
            for h in range(self.height):
                self.weights[:, h, w] += \
                    self.neighbourhood_function.value(BMU, (h, w), epoch) * \
                    self.learning_rate_function.value(epoch) * \
                    (obs - self.weights[:, h, w])
        
    def train(self, X, epochs):
        data = copy.deepcopy(X)
        for epoch in range(self.n_epochs, self.n_epochs + epochs):
             self.random_generator.shuffle(data, axis=0)
             for obs in data:
                 BMU = self.find_BMU(obs)
                 self.__update_weights(BMU, obs, epoch)
        self.n_epochs += epochs
                 
    def plot_map(self, X, Y, colormap='Paired', fig_size=(6, 6)):
        n_classes = len(np.unique(Y))
        results = np.zeros((n_classes, self.height, self.width))
        
        for i in range(len(X)):
            bmu = self.find_BMU(X[i, :])
            results[Y[i], bmu[0], bmu[1]] += 1
        
        def calculate_rates(x):
            if np.all(x == 0):
                return np.zeros_like(x)
            else:
                return x / np.sum(x)
                                 
        rates = np.apply_along_axis(calculate_rates, 0, results)
        winners = np.argmax(rates, axis=0)
        
        # plotting
        fig = plt.figure(figsize=fig_size) 
        ax = fig.add_subplot(111)
        
        color_generator = plt.cm.get_cmap(colormap, n_classes)
        
        for w in range(self.width):
            for h in range(self.height):
                colour = winners[h, w]
                ax.add_patch( 
                    Rectangle((w, h),
                              1, 1,
                              fill=True, color = color_generator(colour), alpha = rates[colour, h, w]) 
                )
                
        
        ax.axis([-0.5, self.width + 0.5, -0.5, self.height + 0.5])
        ax.axis('equal')
        ax.legend(handles=[
                Patch(color=color_generator(i), label=str(i)) 
                for i in range(n_classes)
        ], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('SOM reults')
        plt.gca().invert_yaxis()
        plt.axis('off')
          
        plt.show()
        
        
        
    
