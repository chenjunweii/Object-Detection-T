import os
import tvm
import nnvm
import numpy as np
import mxnet as mx
from mxnet import nd
from tvm import autotvm
from nnvm import compiler
from tvm.contrib import graph_runtime
from time import time

target = 'cuda -libs=cudnn,cublas'

target_host = 'llvm -target=aarch64-linux-gnu'

ctx = tvm.gpu(0)

shapes = (1, 3, 300, 300)

net, args, auxs = mx.model.load_checkpoint('deploy_ssd_mobilenet_v2_300', 1)
 
args.update(auxs)

net, params = nnvm.frontend.from_mxnet(net, args)

inputs = np.ones(shapes)

graph, lib, params = compiler.build(net, target, {"data": shapes}, params = params, target_host = target_host)

print('[*] Model is Compiled')

m = graph_runtime.create(graph, lib, ctx)

print('[*] Graph RunTime is Created')

m.set_input(**params)

print('[*] Run ')

for i in range(10):

    m.set_input('data', tvm.nd.array(inputs.astype('float32'))) # astype

    start = time()

    m.run()

    out = m.get_output(0).asnumpy()

    e = time() - start

    print('Time Cost : ', e)

