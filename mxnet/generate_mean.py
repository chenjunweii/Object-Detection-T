import mxnet as mx
import numpy as np



r = 123

g = 117

b = 104

mean = np.array([r, g, b]).reshape([1, 3, 1, 1])

mean = mx.nd.array(mean)

means = dict()

means['mean'] = mean

mx.nd.save('mean.nd', means)



