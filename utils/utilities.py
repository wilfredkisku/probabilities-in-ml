from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack, exp, asarray
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

import numpy as np

def bimodalDist():
    sample1 = normal(loc = 20, scale = 5, size = 300)
    sample2 = normal(loc = 40, scale = 5, size = 700)

    sample = hstack((sample1, sample2))

    model = KernelDensity(bandwidth = 2, kernel = 'gaussian')
    sample = sample.reshape((len(sample), 1))
    model.fit(sample)

    values = asarray([value for value in range(1, 60)])
    values = values.reshape((len(values), 1))

    probabilities = model.score_samples(values)
    probabilities = exp(probabilities)


    pyplot.hist(sample, bins = 60, density = True)
    pyplot.plot(values[:], probabilities)
    pyplot.show()

    return None
    
def meshPlot():
    mu_x = 0
    var_x = 10.

    mu_y = 0
    var_y = 10.

    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal([mu_x, mu_y],[[var_x,0],[0,var_y]])

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    pyplot.show()
    return None

if __name__ == '__main__':
    
    meshPlot()
    bimodalDist()
