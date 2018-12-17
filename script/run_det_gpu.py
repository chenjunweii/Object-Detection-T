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
from time import time

tgt_host = "llvm"

# Change it to respective GPU if gpu is enabled Ex: cuda, opencl

tgt = "cuda"

ctx = tvm.context(tgt, 0)

target = tvm.target.cuda()

shapes = (1, 3, 512, 512)

net, params = load_mxnet_model('deploy_ssd_inceptionv3_512-det', 215, 'model')

net, params = nnvm.frontend.from_mxnet(net, params)

input_shape = (1, 3, 512, 512)

inputs = np.ones(shapes)

print("[*] Compile...")
#with autotvm.apply_history_best('ssd-inceptionv3.log'):
with compiler.build_config(opt_level = 3):
        graph, lib, params = compiler.build(net, target, {"data": shapes}, params = params)

lib.export_library('so/{}.tvm.so'.format('ssd-inceptionv3-512-det'))

print('[*] Model is Compiled')

m = graph_runtime.create(graph, lib, ctx)

save_tvm_params('params/ssd-inceptionv3-512-det', params)

save_tvm_graph('graph/ssd-inceptionv3-512-det', graph)

print('[*] Graph RunTime is Created')

m.set_input('data', tvm.nd.array(inputs.astype(np.float32)))

m.set_input(**params)

print('[*] Run ')


for i in range(10):
    start = time()
    m.set_input('data', tvm.nd.array(inputs.astype(np.float32)))
    m.run()
    cls_prob = m.get_output(0)
    loc_preds = m.get_output(1)
    anchor_boxes = m.get_output(2)
    print('time : ', time() - start)
#save_tvm_params('test', params)

#save_tvm_graph('test', graph)

#tvm.save_json(lib)

#print(tvm_output)

print(cls_prob.asnumpy().shape)
print(loc_preds.asnumpy().shape)
print(loc_preds.asnumpy().shape)
print(anchor_boxes.asnumpy().sum())




