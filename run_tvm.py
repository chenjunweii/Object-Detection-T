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
target = tvm.target.cuda()
# tvm module for compiled functions.
loaded_lib = tvm.module.load("test.tvm.so")
#loaded_lib = tvm.module.load("test.tvm.tar")
#print(loaded_lib)
#net, params, input_shape = load_model('test') # load mxnet model
#net, params = nnvm.frontend.from_mxnet(net, params)
# json graph
#loaded_json = open(("test.tvm.json")).read()

loaded_json = open(("test.tvm.json")).read()
# parameters in binary
loaded_params = bytearray(open("test.tvm.params", "rb").read())

fcreate = tvm.get_global_func("tvm.graph_runtime.create")

ctx = tvm.gpu(0)

gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
#gmodule = fcreate(graph, lib, ctx.device_type, ctx.device_id)

set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

inputs = np.zeros([1, 3, 224, 224])

#print(tvm.nd.array(inputs.astype(np.int8)).dtype)

set_input("data", tvm.nd.array(inputs.astype(np.float32)))

gmodule["load_params"](loaded_params)

run()

out = tvm.nd.empty([1, 7, 222, 222])

get_output(0, out)

print(out.asnumpy())
