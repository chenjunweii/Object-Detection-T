import numpy as np
import nnvm
import nnvm.symbol as sym
from nnvm.testing.check_computation import check_function


x = sym.Variable('x')

r = sym.Variable('r')

a = sym.Variable('a')

y = sym.repeat(x, r, a)

def forward(x):
    return np.repeat(x)

def backward(head_grads, x):
    return [1. / x * head_grads]

dtype = dict()
dtype['x'] = "float32"
dtype['r'] = 'int32'
dtype['a'] = 'int32'
shape = dict()
shape['x'] = (1, 3)
shape['r'] = (1)
shape['a'] = (1)

values = dict()

values['x'] = np.zeros([1,3])
values['r'] = np.zeros([1])
values['a'] = np.zeros([1])
check_function(y, forward, None, values = values)#, dtype=dtype, shape=shape)
