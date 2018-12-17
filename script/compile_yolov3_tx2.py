import sys
sys.path[0] += '/../'

import nnvm
import nnvm.frontend.darknet
import nnvm.testing.yolo_detection
import nnvm.testing.darknet
import matplotlib.pyplot as plt
import numpy as np
import tvm
<<<<<<< HEAD
from tvm import autotvm
=======
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
import sys

from ctypes import *
from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__
from time import time

from tool.utils import *
from utils import *
# Model name
MODEL_NAME = 'yolov3'

dtype = 'float32'
batch_size = 1

print("Converting darknet to nnvm symbols...")

<<<<<<< HEAD
target = 'cuda'
=======
target = 'cuda -libs=cudnn'
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e

target_host = 'llvm -device=arm_cpu -target=aarch64-linux-gnu'

ctx = tvm.gpu(0)

data = np.empty([batch_size, 3, 608, 608], dtype)

shape = {'data': data.shape}

<<<<<<< HEAD
=======
#dtype_dict = {'data' : 'float32', 'reshape1_mask': 'float32', 'reshape1_bias': 'float32',
#        'reshape1_attr': 'float32', 
#        'reshape3_bias': 'float32', 'reshape3_attr': 'float32', 'reshape3_mask': 'float32',
#        'reshape5_mask': 'float32', 'reshape5_bias': 'float32', 'reshape5_attr' : 'float32' }

>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
dtype_dict = {}

print("Compiling the model...")

out = 'yolov3.tx2.gpu'

ins = 'yolov3.x86.gpu'

graph = load_tvm_graph('graph/{}'.format(ins))

params = load_tvm_params('params/{}'.format(ins))

symbol = graph.symbol

[neth, netw] = shape['data'][2:] # Current image shape is 608x608

<<<<<<< HEAD
with autotvm.apply_history_best('yolov3-darknet.tx2.gpu.log'):
    with nnvm.compiler.build_config(opt_level = 2):
        graph, lib, params = nnvm.compiler.build(symbol, target, shape, dtype = dtype_dict, params = params)
=======
with nnvm.compiler.build_config(opt_level = 2):

    graph, lib, params = nnvm.compiler.build(symbol, target, shape, dtype = dtype_dict, params = params)
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e

######################################################################
# Load a test image
# --------------------------------------------------------------------
test_image = 'dog.jpg'
print("Loading the test image...")
img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + \
          test_image + '?raw=true'
download(img_url, test_image)

data = nnvm.testing.darknet.load_image(test_image, netw, neth)
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime

m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
print("Running the test image...")

# get outputs
tvm_out = []


<<<<<<< HEAD
start = time()
m.run()
for i in range(3):
    layer_out = {}
    layer_out['type'] = 'Yolo'
    layer_attr = m.get_output(i*4+3).asnumpy()
=======
for i in range(3):
    start = time()
    layer_out = {}
    layer_out['type'] = 'Yolo'
    # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
    m.run()
    layer_attr = m.get_output(i*4+3).asnumpy()

    print(layer_attr.shape)

    print('cost : ', time() - start)
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
    layer_out['biases'] = m.get_output(i*4+2).asnumpy()
    layer_out['mask'] = m.get_output(i*4+1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    tvm_out.append(layer_out)

<<<<<<< HEAD
print('cost : ', time() - start)
=======
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
# do the detection and bring up the bounding boxes
thresh = 0.5
nms_thresh = 0.45
img = nnvm.testing.darknet.load_image_color(test_image)
_, im_h, im_w = img.shape
dets = nnvm.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                      1, tvm_out)
last_layer = net.layers[net.n - 1]
nnvm.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = 'coco.names'
coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name + '?raw=true'
font_name = 'arial.ttf'
font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name + '?raw=true'
download(coco_url, coco_name)
download(font_url, font_name)

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

nnvm.testing.yolo_detection.draw_detections(img, dets, thresh, names, last_layer.classes)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
