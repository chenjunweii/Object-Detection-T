import sys
sys.path[0] += '/../'

import os
import tvm
import cv2
import nnvm
import zipfile
import numpy as np
from mxnet import nd
from tvm import autotvm
from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint
from tool.load_deploy_model import * # import load_deploy_model # import save_tvm_params, save_tvm_graph
from utils import *
from time import time

tgt_host = "llvm"

ctx = tvm.gpu(0)

target = 'cuda -libs=cudnn'

shapes = (1, 712, 712, 3)

net, params = load_mxnet_model('deploy_ssd_mobilenet_v2_680-det', 240, 'model')

r, g, b = 123, 117, 104 

params['mean'] = nd.array(np.array([r, g, b]).astype(np.float32).reshape([1, 3, 1, 1]))

net, params = nnvm.frontend.from_mxnet(net, params)

print("[*] Compile...")

#with autotvm.apply_history_best('log/ssd-inceptionv3.log'):
with compiler.build_config(opt_level = 3):
    graph, lib, params = compiler.build(net, target, {"data": shapes, "mean" : (1, 3, 1, 1)}, dtype = 'float32')

out = 'ssd-mobilenetv2-712-det'

lib.export_library('so/{}.tvm.so'.format(out))

print('[*] Model is Compiled')

m = graph_runtime.create(graph, lib, ctx)

#save_tvm_params('params/{}'.format(out), params)

#save_tvm_graph('graph/{}'.format(out), graph)

print('[*] Graph RunTime is Created')

inputs = np.zeros(shapes).astype(np.float32)

m.set_input('data', tvm.nd.array(inputs)) # astype

#m.set_input(**params)

print('[*] Run ')

total = 0

for i in range(10):

    #ndinputs
    #start = time()
    #e = time() - start
    #total += e
    #print('set input : ', e)
    start = time()
    m.run()
    e = time() - start
    total += e
    print('Forward : ', e)

    start = time()
    cls_prob = m.get_output(0).asnumpy()
    e = time() - start
    total += e
    print('get class prob : ', e)

    start = time()
    loc_preds = m.get_output(1).asnumpy()
    e = time() - start
    total += e
    print('get loc preds : ', e)

    start = time()
    
    anchor_boxes = m.get_output(2).asnumpy()
    
    e = time() - start
    
    total += e

    print('get anchor_boxes : ', e)

    print('Total : ', total)

    print('anchor sum : ', anchor_boxes.sum())
    
    print('=========================')

    total = 0

print(cls_prob.asnumpy().shape)
print(loc_preds.asnumpy().shape)
print(loc_preds.asnumpy().shape)




