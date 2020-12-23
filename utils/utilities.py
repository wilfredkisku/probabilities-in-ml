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

def pdf(data, mean: float, variance: float):
    s = 1/(np.sqrt(2*np.pi*variance))
    s = s * (np.exp(-(np.square(data - mean)/(2*variance))))
    return s
def pdfClustering():
    k = 3
    weights = np.ones((k)) / k
    #k value will give the base for the cluster number
    means = np.random.choice(X, k)
    variances = np.random.random_sample(size=k)
    
    X = np.array(X)

def gaussianModels():
    n_samples = 200
    mu1, sigma1 = -4, 1.2
    mu2, sigma2 =  4, 1.8
    mu3, sigma3 =  0, 1.6

    x1 = np.random.normal(mu1, np.sqrt(sigma1), n_samples)
    x2 = np.random.normal(mu2, np.sqrt(sigma2), n_samples)
    x3 = np.random.normal(mu3, np.sqrt(sigma3), n_samples)

    print(x1)
    print(x2)
    print(x3)

    X = np.array(list(x1) + list(x2) + list(x3))
    np.random.shuffle(X)
    print(X.shape)
    bins = np.linspace(np.min(X), np.max(X), 200)

    pyplot.figure(figsize=(10,7))
    pyplot.xlabel("$x$")
    pyplot.ylabel("pdf")
    pyplot.scatter(X, [0.005] * len(X), color='green', s=30, marker=2, label="Train data")

    pyplot.plot(bins, pdf(bins, mu1, sigma1), color='red', label="True pdf")
    pyplot.plot(bins, pdf(bins, mu2, sigma2), color='red')
    pyplot.plot(bins, pdf(bins, mu3, sigma3), color='red')

    pyplot.legend()
    pyplot.plot()
    pyplot.show()
    return None

if __name__ == '__main__':
    
    #meshPlot()
    #bimodalDist()
    gaussianModels()
    #pdfClustering()
