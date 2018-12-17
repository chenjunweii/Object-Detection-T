import tvm
import nnvm
from tvm.contrib import graph_runtime
import numpy as np
from load_deploy_model import load_model
import os
import zipfile
import tvm
import mxnet as mx
import cv2
import numpy as np
#from pudb import set_trace; set_trace()

from tvm import autotvm
from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib.download import download
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint
from utils import save_tvm_params, save_tvm_graph
from load_deploy_model import load_mxnet_model

tgt_host = "llvm"

# Change it to respective GPU if gpu is enabled Ex: cuda, opencl

tgt = "cuda"

ctx = tvm.context(tgt, 0)

target = tvm.target.cuda()

shapes = (1, 3, 512, 512)

net, params = load_mxnet_model('deploy_ssd_inceptionv3_512', 215, 'model')

net, params = nnvm.frontend.from_mxnet(net, params)

input_shape = (1, 3, 512, 512)

inputs = np.ones(shapes)

#print(net.debug_str())

nodes = (net.get_internals())

print(net.debug_str())

#nets = (net.get_children().list_output_names())
nets = (net.list_output_names())

print('All Outputs : ', nets)

print("[*] Compile...")
with autotvm.apply_history_best('ssd-inceptionv3.log'):
    with compiler.build_config(opt_level = 3):
        for net in nets:
            print('building graph for {}'.format(net))
            graph, _, _ = compiler.build(nodes[net], target, {"data": shapes}, params = params)
            print('{} is done'.format(net))
raise

m = graph_runtime.create(graph, lib, ctx)

m.set_input('data', tvm.nd.array(inputs.astype(np.float32)))

m.set_input(**params)

m.run()

tvm_output = m.get_output(0)

#save_tvm_params('test', params)

#save_tvm_graph('test', graph)

#tvm.save_json(lib)

#print(tvm_output)

#print(type(tvm_output))

print(tvm_output.asnumpy().sum())

print(tvm_output.asnumpy().shape)



