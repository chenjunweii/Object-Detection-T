import caffe
from caffe.model_libs import *
from caffe import layers as L
from caffe import params as P

#net = caffe.Net('ssd.prototxt', caffe.TEST) #, 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', caffe.TEST)
#-net = caffe.Net('inceptionv3.prototxt', caffe.TEST) #, 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', caffe.TEST)


def multi_layer_feature(net, from_layers, num_filters, strides, pads, min_filter = 128):
    use_batchnorm = True
    use_relu = True
    assert len(from_layers) > 0 
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            layers.append(from_layer)
        else:
            assert (len(layers) > 0)
            assert (num_filter > 0)
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            ConvBNLayer(net, layer, 'multi_feat_%d_conv_1x1', use_batchnorm, use_relu, num_filter, 1, 0, 1, lr_mult = lr_mult)
            ConvBNLayer(net, layer, 'multi_feat_%d_conv_3x3', use_batchnorm, use_relu, num_filter, 3, p, s, lr_mult = lr_mult)
            layers.append(layer)
            #conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k), num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            #conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k), num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu') 
            #layers.append(conv_3x3)
    return layers

net = caffe.NetSpec()

net.data = L.ImageData(name = 'data')

InceptionV3Body(net, from_layer = 'data', fully_conv = True, reduced = True, dilated = True, dropout = False)

net = caffe.NetSpec()

test_data = "examples/VOC0712/VOC0712_test_lmdb"
test_batch_size = 8
label_map_file = "data/VOC0712/labelmap_voc.prototxt"
#net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
#        train=False, output_label=True, label_map_file=label_map_file,
#        transform_param=test_transform_param)


from_layers = ['mixed_7/join', 'mixed_10/join', '', '', '', ''] 
#from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', ''] 
num_filters = [-1, -1, 512, 256, 256, 128]
strides = [-1, -1, 2, 2, 2, 2]
pads = [-1, -1, 1, 1, 1, 1]
sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]

aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
normalizations = -1

#net.data = L.Data(name = 'data')

net.data = L.Layer()

InceptionV3Body(net, from_layer = 'data', fully_conv = True, reduced = True, dilated = True, dropout = False)

min_sizes = [m[0]for m in sizes]

max_sizes = [m[1]for m in sizes]

steps = [8, 16, 32, 64, 100, 300]
use_batchnorm = False

lr_mult = 1

mbox_source_layers = multi_layer_feature(net, from_layers, num_filters, strides, pads, min_filter = 128)

#mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

conf_name = "mbox_conf"
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

#shutil.copy(test_net_file, job_dir)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
shutil.copy(deploy_net_file, job_dir)
