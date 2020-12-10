from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack, exp, asarray
from sklearn.neighbors import KernelDensity

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
