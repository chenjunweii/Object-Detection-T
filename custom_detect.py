import os
import cv2
import time
import mxnet as mx
import numpy as np
from detect.detector import Detector
from custom_visualize import visualize_detection


#out = mx.sym.load('model/deploy_ssd_mobilenet_v2_300-symbol.json')

out, arg_params, aux_params = mx.model.load_checkpoint('model/deploy_ssd_mobilenet_v2_300', 150) # must be deploy net

out = mx.sym.load('model/deploy_ssd_mobilenet_v2_300-symbol.json')

#batch_shape = (1, 3, 300, 300)

#out, arg_params, aux_params = mx.model.load_checkpoint('model/deploy_ssd_inceptionv3_512', 240) # must be deploy net

#out = mx.sym.load('model/deploy_ssd_inceptionv3_512-symbol.json')

batch_shape = (1, 3, 300, 300)

ctx = mx.gpu(0)

all_params = dict([(k, v.as_in_context(ctx)) for k, v in arg_params.items()])


"""
os.environ['MXNET_USE_TENSORRT'] = '1'

executor = mx.contrib.tensorrt.tensorrt_bind(out,
                                            ctx = ctx,
                                            all_params = all_params,
                                            data = batch_shape,
                                            grad_req = 'null',
                                            force_rebind = True)

executor.copy_params_from(arg_params, aux_params)
#print(out)

"""
os.environ['MXNET_USE_TENSORRT'] = '0'

executor = out.simple_bind(ctx = ctx,
                           data = batch_shape,
                           grad_req = 'null',
                           force_rebind = True)

executor.copy_params_from(arg_params, aux_params)

#"""

inputs = mx.nd.zeros(batch_shape)

print('Warming up TensorRT')

for i in range(0, 10):
    
    y_gen = executor.forward(is_train = False, data = inputs)
    
    y_gen[0].wait_to_read()

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', \
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', \
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


# Timing

print('Starting TensorRT timed run')

frame = cv2.imread('data/demo/dog1.jpg')

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

print('default : ', frame.dtype)

#frame = frame.astype(np.float32)

mean_pixels = (123, 117, 104)

mean_pixels_nd = mx.nd.array(mean_pixels, ctx = ctx).reshape((3,1,1))

for i in range(10):

    frame_resize = mx.nd.array(cv2.resize(frame, (300, 300)), ctx = ctx)
    
    #frame_resize = mx.img.imresize(frame, self.data_shape[0], self.data_shape[1], cv2.INTER_LINEAR)
    # Change dimensions from (w,h,channels) to (channels, w, h)
    
    frame_t = mx.nd.transpose(frame_resize, axes = (2, 0, 1))
    
    frame_norm = frame_t - mean_pixels_nd
    
    # Add dimension for batch, results in (1,channels,w,h)
    
    #batch_frame = [mx.nd.expand_dims(frame_norm, axis = 0)]

    batch_frame = frame_norm.expand_dims(0)
    
    #input = mx.nd.random_normal(shape = input.shape, ctx = ctx)
    
    #mean_pixels = (args.mean_r, args.mean_g, args.mean_b)
    start = time.time()

    y_gen = executor.forward(is_train = False, data = batch_frame)

    y_gen[0].wait_to_read()

    #print(y_gen[0].asnumpy())

    print('Time : ', time.time() - start)

    """
    #frame_resize = mx.nd.array(cv2.resize(frame, (self.data_shape[0], self.data_shape[1])))
    
    #frame_resize = mx.img.imresize(frame, self.data_shape[0], self.data_shape[1], cv2.INTER_LINEAR)
    
    
    # Change dimensions from (w,h,channels) to (channels, w, h)
    
    #frame_t = mx.nd.transpose(frame_resize, axes=(2,0,1))
    
    #frame_norm = frame_t - self.mean_pixels_nd
    
    print(y_gen[0].asnumpy().shape)
    """ 

    print(y_gen[0].asnumpy().flatten()[:20])

    #result = Detector.filter_positive_detections(y_gen[0].asnumpy())

    #for k, det in enumerate(result):

        #img = cv2.imread(im_list[k])

        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #    visualize_detection(frame, det, classes, 0.6)

    #print(time.process_time() - start)
