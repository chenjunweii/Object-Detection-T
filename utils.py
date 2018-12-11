import tvm
import nnvm

def load_tvm_module(filename, ext = 'so'):

    filename = '{}.tvm.{}'.format(filename, ext)

    tnds = tvm.module.load(filename)

    return tnds
def load_tvm_params(filename):

    filename = '{}.tvm.params'.format(filename)

    byte = bytearray(open(filename, 'rb').read())

    tnds = nnvm.compiler.load_param_dict(byte)

    return tnds

def save_tvm_params(network, params):

    fn = '{}.tvm.params'.format(network)
    tnds_byte = nnvm.compiler.save_param_dict(params)
    
    File = open(fn, "wb")
    
    File.write(tnds_byte)

    print('[*] TVM Parameters is save to {}'.format(fn))

def save_tvm_graph(network, graph):

    fn = '{}.tvm.json'.format(network)

    with open(fn, "w") as fo:
        
        fo.write(graph.json())
    
    print('[*] TVM Graph is save to {}'.format(fn))


def load_tvm_graph(network):

    fn = '{}.tvm.json'.format(network)
    
    print('[*] TVM Graph is loaded {}'.format(fn))

    return nnvm.graph.load_json(open(fn, 'r').read())

    return nnvm.graph.load_json(fn)
