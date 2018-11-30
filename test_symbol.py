import numpy as np
import mxnet as mx


data = mx.symbol.Variable('data')

out = mx.sym.Convolution(data, kernel = (3,3), num_filter = 7)

nds = dict()

ctx = mx.cpu()

nds['data'] = mx.nd.array(np.ones([1, 3, 224, 224]), ctx)
nds['convolution0_weight'] = mx.nd.array(np.ones([7, 3, 3, 3]), ctx)
nds['convolution0_bias'] = mx.nd.array(np.ones([7]), ctx)

e = out.bind(mx.cpu(), nds)

e.forward(False)

out.save('test.json')

mx.nd.save('test.params', nds)


