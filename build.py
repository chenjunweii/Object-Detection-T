
import nnvm
import numpy as np
from nnvm import testing
from tvm import autotvm
from tvm.contrib.util import tempdir
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tool.load_deploy_model import load_model, load_mxnet_model
from utils import *

def get_network(name, batch_size, input_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif name == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'squeezenet_v1.1':
        net, params = nnvm.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1')
    elif name == 'inception_v3':
        #input_shape = (1, 3, 512, 512)
        net, params = nnvm.testing.inception_v3.get_workload(batch_size=batch_size)

    elif name == 'ssd-mobilenetv2':

        net, params = load_mxnet_model('deploy_ssd_mobilenet_v2_680-det', 240, 'model')

        net, params = nnvm.frontend.from_mxnet(net, params)

        input_shape = (1, input_size, input_size, 3)
        input_shape = (1, 680, 680, 3)

        output_shape = None

    elif name == 'ssd-inceptionv3-nms':

        net, params = load_mxnet_model('deploy_ssd_inceptionv3_512-det', 215, 'model')

        net, params = nnvm.frontend.from_mxnet(net, params)

        input_shape = (1, 512, 512, 3)

        output_shape = None

    elif name == 'ssd-inceptionv3':

        net, params = load_mxnet_model('deploy_ssd_inceptionv3_512-det', 215, 'model')

        net, params = nnvm.frontend.from_mxnet(net, params)

        input_shape = (1, input_size, input_size, 3)
        input_shape = (1, 512, 512, 3)

        output_shape = None

    elif name == 'ssd-inceptionv3-fp16':

        net, params = load_mxnet_model('deploy_ssd_inceptionv3_fp16_512-det', 215, 'model')

        net, params = nnvm.frontend.from_mxnet(net, params)

        input_shape = (1, 512, 512, 3)

        output_shape = None


    elif name == 'yolov3-darknet':

        ins = 'yolov3.x86.gpu'

        graph = load_tvm_graph('graph/{}'.format(ins))

        params = load_tvm_params('params/{}'.format(ins))

        net = graph.symbol

        input_shape = (1, 3, 608, 608)
        
        output_shape = None

    elif name == 'yolov3-darknet-320':

        ins = 'yolov3-darknet-320.x86.gpu'

        graph = load_tvm_graph('graph/{}'.format(ins))

        params = load_tvm_params('params/{}'.format(ins))

        net = graph.symbol

        input_shape = (1, 3, 320, 320)
        
        output_shape = None
    
    elif name == 'yolov3-darknet-tiny':

        ins = 'yolov3-darknet-tiny.x86.gpu'

        graph = load_tvm_graph('graph/{}'.format(ins))

        params = load_tvm_params('params/{}'.format(ins))

        net = graph.symbol

        input_shape = (1, 3, 224, 224)
        
        output_shape = None

    elif name == 'yolov3-darnknet':

        net, params = load_mxnet_model('yolo3_darknet53_voc', 0, 'model')

        net, params = nnvm.frontend.from_mxnet(net, params)

        input_shape = (1, 3, 416, 416)

        output_shape = None
    
    elif name == 'custom':
        # an example for custom network
        from nnvm.testing import utils
        net = nnvm.sym.Variable('data')
        net = nnvm.sym.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        net = nnvm.sym.flatten(net)
        net = nnvm.sym.dense(net, units=1000)
        net, params = utils.create_workload(net, batch_size, (3, 224, 224))
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = nnvm.frontend.from_mxnet(block)
        net = nnvm.sym.softmax(net)


    elif name == 'test_nms':
        
        net, params, input_shape = load_model('test_nms')
        net, params = nnvm.frontend.from_mxnet(net)

    elif name == 'test':
        
        net, params, input_shape = load_model('test')
        net, params = nnvm.frontend.from_mxnet(net)

    else:
    
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape
