# Principal Component Analysis 

# Written by Pin-Chih Su

# Tested in python 2.7.6, numpy 1.9.0, scipy-0.14.0, matplotlib.pyplot-1.3.1, sklearn 0.15.2

import pylab
from pylab import *
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA

# Create some 2D data points
mean   = array([2.11, -5.3])
covariance_matrix = matrix([[0.6, -0.3], [-0.3, 1.8]])
nrvt   = multivariate_normal(mean, covariance_matrix, 1000)
 
# Normalize the data
nrvt = (nrvt - mean)/std(nrvt, axis=0)

# PCA transformation

pca = PCA(n_components=2)

Y = pca.fit(nrvt).transform(nrvt)

############## Plot results

## Define plots:
def plot_samples(sample, axis_list=None):
    plt.scatter(sample[:, 0], sample[:, 1], s=2, marker='.',color='black', linewidths=2, zorder=10,alpha=1)
    if axis_list is not None:
        colors = ['orange']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color=color)

    # Horizontal and vertical grid lines in plots
    plt.hlines(0, -4, 4)
    plt.vlines(0, -4, 4)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')

# Plot the original data plus the principal components 
axis_list = [pca.components_.T]         
plt.subplot(2, 2, 1)                                    # Sub-plot location
plt.title("Original Data")                              # plot title
plot_samples(nrvt, axis_list=axis_list)                 # Activate principal component plotting here
legend = plt.legend(['PCA'], loc='upper right')         # Principal component legends
legend.set_zorder(10)


#After PCA transformed plot
plt.subplot(2, 2, 2)                                    # Sub-plot location
plt.title("PCA")                                        # plot title
plot_samples(Y, axis_list=None)             
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36) # Adjust subplot arrangement
plt.savefig("original_pca.jpg")                         # save the plot

plt.clf()
