import os
import tvm
import nnvm
import numpy as np
from nnvm import testing
from matplotlib import pyplot as plt
from tvm import autotvm
from tvm.contrib.util import tempdir
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import tvm.contrib.graph_runtime as runtime

from build import get_network
from utils import save_tvm_graph, save_tvm_params, load_tvm_graph, load_tvm_params

from gluoncv import model_zoo, data, utils

from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata


im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                              'gluoncv/detection/street_small.jpg?raw=true',
                              'street_small.jpg', module='data')


model = 'yolo3_darknet53_coco_gluoncv_block'

target = tvm.target.cuda()

size = 320

x, img = data.transforms.presets.ssd.load_test(im_fname, short=size)

board = None

device = 'gpu'

#log_filename = '{}-{}.{}.{}.log'.format(model, size, board, device)
log_filename = '{}.{}.{}.log'.format(model, board, device)


with autotvm.apply_history_best(log_filename):

    loaded_lib = tvm.module.load("lib/{}.tvm.so".format(model))

    loaded_json = open(("graph/{}.tvm.json".format(model))).read()
    # parameters in binary
    loaded_params = (bytearray(open("params/{}.tvm.params".format(model), "rb").read()))

    #nnvm.compiler.load_param_dict(loaded_params)

    fcreate = tvm.get_global_func("tvm.graph_runtime.create")

    ctx = tvm.gpu(0)
    
    #module = runtime.create(loaded_json, loaded_lib, ctx)

    gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)

    set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]

    #inputs = np.zeros([1, 3, size, size])

    #print(tvm.nd.array(inputs.astype(np.int8)).dtype)

    #inputs = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

    set_input("data", tvm.nd.array(x.asnumpy().astype(np.float32)))
            
    #module.set_input(**loaded_params)

    #print(loaded_params)

    gmodule["load_params"](loaded_params)

    run()
    
    block = model_zoo.yolo3_darknet53_coco(pretrained_base=True, pretrained=True)

    #class_IDs, scores, bounding_boxs = run(loaded_json, loaded_lib, loaded_params, ctx)

    class_IDs = get_output(0)

    scores = get_output(1)

    bounding_boxs = get_output(2)

    ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=block.classes)
    plt.show()

