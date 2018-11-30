import argparse 
import os
import tvm
import nnvm
import numpy as np
from nnvm import testing
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime
from load_deploy_model import load_mxnet_model
from tuner import tuner

parser = argparse.ArgumentParser(description = '')
 
parser.add_argument('--network', type = str, default = None, help = 'Network Architecture')

parser.add_argument('--target', type = str, default = 'cuda', help = 'Deploy Target')

parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data Type')

parser.add_argument('--tuner', type = str, default = 'xgb', help = 'Select Tuner')

args = parser.parse_args()

if args.target == 'cuda':

    target = tvm.target.cuda()

else:

    print('[!] Not Supported Yet')


option = {
    'network' : args.network,
    'dtype' : args.dtype,
    'target' : target,
    'log_filename': '{}.log'.format(args.network),
    'tuner': args.tuner,
    'n_trial': 2000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder = autotvm.LocalBuilder(timeout = 10),
        runner = autotvm.LocalRunner(number = 20, repeat = 3, timeout = 4),
    ),
}

T = tuner(**option)

T.tune()








