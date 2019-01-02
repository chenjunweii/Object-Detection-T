
#net = caffe.Net('.prototxt', 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', caffe.TEST)

"""
def add_conv_act():
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
    stride=stride, num_filter=num_filter, name="{}_conv".format(name))
    if use_batchnorm:
    conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
    name="{}_{}".format(name, act_type))
    return rel

"""


def add_bn():

    pass

def add_fc():

    s = '''
layer {{
    name: "{}"
    type: "Convolution"
    bottom: "{}"
    top: "{}"
    convolution_param {{
        num_output: {}
        pad: {}
        kernel_size : {}
        dilation: {}
        weight_filler {{type: "xavier"}}
        bias_filler {{
            type: "constant"
            value: 0
        }}
    }}
}}
    '''.format(n, i, o, num_filter, pad, kernel_size, dilation)

def add_reshape():

    pass

def add_flatten():

    pass


def add_conv(f, n, i, o, num_filter, kernel_size, pad, dilation = 0):

    s = '''
layer {{
    name: "{}"
    type: "Convolution"
    bottom: "{}"
    top: "{}"
    convolution_param {{
        num_output: {}
        pad: {}
        kernel_size : {}
        dilation: {}
        weight_filler {{type: "xavier"}}
        bias_filler {{
            type: "constant"
            value: 0
        }}
    }}
}}
    '''.format(n, i, o, num_filter, pad, kernel_size, dilation)

    #print(s.format(n, i, o, num_filter, pad, kernel_size, dilation))

    f.write(s)

    print(s)

def read_prototxt(filename):

    f = open(filename, 'r')

    return f

def create_prototxt(filename):

    f = open(filename, 'w+')

    return f

def copy_prototxt(i, o):

    fi = read_prototxt(i)

    fo = create_prototxt(o)

    fo.writelines(fi.readlines())

    fi.close()

    return fo

f = copy_prototxt('test.prototxt', 'test2.prototxt')

add_conv(f, "conv1_1", "data", "conv1_1", 64, 1, 3)



f.close()







"""

def add_extra_layer():

    num_1x1 = max(min_filter, num_filter // 2)
    conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
    num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
    conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k),
    num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu')
    layers.append(conv_3x3)

def add_detection():

    pass

"""

#add_extra_layer('')


