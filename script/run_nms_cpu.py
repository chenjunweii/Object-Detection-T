import tvm
import nnvm
from tvm.contrib import graph_runtime
import numpy as np
from load_deploy_model import load_model, load_nms
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
from time import time

tgt_host = "llvm"

# Change it to respective GPU if gpu is enabled Ex: cuda, opencl

tgt = "llvm"

ctx = tvm.context(tgt, 0)

target = 'llvm'

shapes = dict()

shapes['cls_prob'] = (1, 21, 5186)
shapes['loc_preds'] = (1, 20744)
shapes['anchor_boxes'] = (1, 5186, 4)

net = load_nms('model/deploy_ssd_inceptionv3_512-nms-symbol')

net, params = nnvm.frontend.from_mxnet(net)

print("[*] Compile...")

#with autotvm.apply_history_best('ssd-inceptionv3.log'):
with compiler.build_config(opt_level = 3):
        graph, lib, params = compiler.build(net, target, shapes)

lib.export_library('so/{}.tvm.so'.format('ssd-inceptionv3-512-nms'))

print('[*] Model is Compiled')

m = graph_runtime.create(graph, lib, ctx)

for s in shapes.keys():

    m.set_input(s, tvm.nd.array(np.zeros(shapes[s]).astype(np.float32)))

start = time()

print(graph)

save_tvm_graph('graph/ssd-inceptionv3-512-nms', graph)

#save_tvm_graph('graph/ssd-inceptionv3-512-nms', params)

m.run()

tvm_output = m.get_output(0)

print(net.debug_str())

print('time : ', time() - start)

#save_tvm_params('test', params)

#save_tvm_graph('test', graph)

#tvm.save_json(lib)

#print(tvm_output)

#print(type(tvm_output))

print(tvm_output.asnumpy().sum())

print(tvm_output.asnumpy().shape)



