import numpy as np
import mxnet as mx


data = mx.symbol.Variable('data')
data1 = mx.symbol.Variable('data1')
data2 = mx.symbol.Variable('data2')

out = mx.sym.contrib.MultiBoxDetection(data, data1, data2)

nds = dict()

ctx = mx.cpu()

nds['data'] = mx.nd.array(np.ones([1, 77, 6]), ctx)
nds['data1'] = mx.nd.array(np.ones([1, 77]), ctx)
nds['data2'] = mx.nd.array(np.ones([1, 77, 6]), ctx)

#nds['valid'] = mx.nd.array(np.ones([7]), ctx)

e = out.bind(mx.cpu(), nds)

e.forward(False)

print(e.outputs[0].shape)

out.save('test_nms.json')

mx.nd.save('test_nms.params', nds)


