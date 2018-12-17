from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib.download import download
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint
import tvm
import nnvm
from load_deploy_model import load_model
import numpy as np
from tvm import autotvm

target = 'llvm'

loaded_lib = tvm.module.load("so/ssd-inceptionv3-512-nms.tvm.so")

loaded_json = open(("graph/ssd-inceptionv3-512-nms.tvm.json")).read()

fcreate = tvm.get_global_func("tvm.graph_runtime.create")

ctx = tvm.cpu(0)

gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)

set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

shapes = dict()
shapes['cls_prob'] = (1, 21, 5186)
shapes['loc_preds'] = (1, 20744) 
shapes['anchor_boxes'] = (1, 5186, 4)

for k, v in shapes.items():

    set_input(k, tvm.nd.array(np.zeros(v).astype(np.float32)))

run()

out = tvm.nd.empty([1, 5186, 6])

get_output(0, out)

print(out.asnumpy())
