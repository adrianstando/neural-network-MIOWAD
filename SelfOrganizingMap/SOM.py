import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, RegularPolygon
from sklearn.manifold import TSNE
from .Utils import transform_coord_to_hex

class SOM:
    def __init__(self, width, height, vector_in_size, neighbourhood_function, learning_rate_function, weight_initializer, hexagon_map=False):
        self.width = width
        self.height = height
        self.vector_in_size = vector_in_size
        self.neighbourhood_function = neighbourhood_function
        self.learning_rate_function = learning_rate_function
        
        self.hexagon_map = hexagon_map
        self.neighbourhood_function.is_hexagon = hexagon_map
        
        self.random_generator = np.random.default_rng()
        
        self.weights = weight_initializer.initialize(width, height, vector_in_size)
        self.n_epochs = 1
    
    def find_BMU(self, x):
        #https://stackabuse.com/self-organizing-maps-theory-and-implementation-in-python-with-numpy/
        distances = (np.square(self.weights - x[..., np.newaxis, np.newaxis])).sum(axis=0)
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
                 
    def plot_map(self, X, Y, type=0, colormap='Paired', fig_size=(6, 6), ax=None, legend=True, dot_size = 100, show_axis = False, invert_y_axis = True , *args_tsne, **kwargs_tsne):
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
        if ax is None:
            fig = plt.figure(figsize=fig_size) 
            ax = fig.add_subplot(111)
            
        color_generator = plt.cm.get_cmap(colormap, n_classes)
        
        if invert_y_axis:
            plt.gca().invert_yaxis()
            
        if type == 0:
            if not self.hexagon_map:
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
                if legend:
                    ax.legend(handles=[
                            Patch(color=color_generator(i), label=str(i)) 
                            for i in range(n_classes)
                    ], loc='center left', bbox_to_anchor=(1, 0.5))
    
            else:
                for w in range(self.width):
                    for h in range(self.height):
                        colour = winners[h, w]
                        ax.add_patch( 
                            RegularPolygon(transform_coord_to_hex((w, h)),
                                           numVertices=6, radius=1,
                                           fill=True, color = color_generator(colour), alpha = rates[colour, h, w]) 
                        )
                        
                
                ax.axis([-np.sqrt(3), self.width * np.sqrt(3) + np.sqrt(3), -2, self.height * 1.5 + 2])
                ax.axis('equal')
                if legend:
                    ax.legend(handles=[
                            Patch(color=color_generator(i), label=str(i)) 
                            for i in range(n_classes)
                    ], loc='center left', bbox_to_anchor=(1, 0.5))
                
            
            
        elif type == 1:                
            points = self.weights.transpose(2, 1, 0).reshape(-1, self.vector_in_size)
            colours = winners.transpose(1, 0).reshape(-1, 1)
            
            alphas = [rates[winners[h, w], h, w] for w in range(self.width) for h in range(self.height)]
            rates_of_winners = alphas
            
            non_zero_rates = []
            for i in range(self.width * self.height):
                if rates_of_winners[i] != 0:
                    non_zero_rates.append(i)
                    
            points = points[non_zero_rates]
            colours = np.array(colours)[non_zero_rates]
            alphas = np.array(alphas)[non_zero_rates]
            
            if self.vector_in_size > 2:
                if points.shape[0] > 1:
                    tsne = TSNE(n_components=2, *args_tsne, **kwargs_tsne)
                    points = tsne.fit_transform(points)
                    plt.scatter(points[:,0], points[:,1], color = color_generator(colours), s=dot_size, alpha=alphas)
                else:
                    plt.text(0, 0, 'Not enough points for t-SNE',
                             horizontalalignment='center',
                             verticalalignment='top', 
                             fontsize=18)
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                
            else:
                plt.scatter(points[:,0], points[:,1], color = color_generator(colours), s=dot_size, alpha=alphas)
                
            if legend:
                ax.legend(handles=[
                        Patch(color=color_generator(i), label=str(i)) 
                        for i in range(n_classes)
                ], loc='center left', bbox_to_anchor=(1, 0.5))
            
        plt.title('SOM reults')
        
        if not show_axis:
            plt.axis('off')
              
        # plt.show()
            
        
        
        
    
