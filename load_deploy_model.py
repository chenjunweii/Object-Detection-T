import os
import mxnet as mx


def load_nms(network):

    return mx.sym.load(network + '.json')

def load_mxnet_model(network, epoch, directory):

    fn = os.path.join(directory, network)

    _, args, auxs = mx.model.load_checkpoint(fn, epoch) # must be deploy net

    network = mx.sym.load(fn + '-symbol.json')
    
    return network, combine_params(args, auxs)

def load_model(network):
    
    if 'nms' in network:

        batch_shape = (1, 77, 6)

        print(network)

    else:

        batch_shape = (1, 3, 224, 224)

    params = mx.nd.load(network + '.params') # must be deploy net

    network = mx.sym.load(network + '.json')

    return network, params, batch_shape

def combine_params(args, auxs):

    args.update(auxs)

    return args

