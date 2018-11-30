import mxnet as mx

def load_mxnet_model(network, epoch, directory):

    fn = os.path.join(directory, network)

    _, arg_params, aux_params = mx.model.load_checkpoint(fn, epoch) # must be deploy net

    network = mx.sym.load(fn + '-symbol.json')

    batch_shape = (1, 3, 512, 512)


def load_model(network):
    
    params = mx.nd.load(network + '.params') # must be deploy net

    network = mx.sym.load(network + '.json')

    batch_shape = (1, 3, 224, 224)

    return network, params, batch_shape

def combine_params(args, auxs):

    pass

