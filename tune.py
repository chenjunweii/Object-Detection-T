import argparse 
import os
import tvm
import nnvm
import numpy as np
from nnvm import testing
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime
from tuner import tuner

parser = argparse.ArgumentParser(description = '')
 
parser.add_argument('--network', type = str, default = None, help = 'Network Architecture')

parser.add_argument('--target', type = str, default = 'cuda', help = 'Deploy Target')

parser.add_argument('--board', type = str, help = 'board')

parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data Type')

parser.add_argument('--tuner', type = str, default = 'xgb', help = 'Select Tuner')

parser.add_argument('--recompile', action = 'store_true', help = 'ReCompile')

parser.add_argument('--local', action = 'store_true', help = 'ReCompile')

parser.add_argument('--remote', action = 'store_true', help = 'ReCompile')

parser.add_argument('--device', type = str, help = 'Select Tuner')

args = parser.parse_args()

target_host = None

if args.target == 'cuda':

    #target = 'cuda -libs=cudnn'#tvm.target.cuda()
    target = 'cuda'

elif args.target == 'llvm':

    target = 'llvm'

else:

    print('[!] Not Supported Yet')

if args.board == 'tx2':

    #target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')

    if args.device == 'cpu':
        target = 'llvm'
    elif args.device == 'gpu':
        target = 'cuda'
    target_host = 'llvm -target=aarch64-linux-gnu'
    device_key = 'tx2'

if args.remote:

    runner = autotvm.RPCRunner(
        device_key,
        host = 'localhost',
        port = 9190,
        number = 5,
        timeout = 4)

elif args.local:

    runner = autotvm.LocalRunner(number = 20, repeat = 3, timeout = 4)

log_filename = '{}.{}.{}.log'.format(args.network, args.board, args.device)

print('[*] Log File : ', log_filename)

option = {
    'recompile' : args.recompile,
    'board' : args.board,
    'target_host' : target_host if target_host != None else 'llvm',
    'network' : args.network,
    'dtype' : args.dtype,
    'target' : target,
    'log_filename': log_filename,
    'tuner': args.tuner,
    'n_trial': 2000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder = autotvm.LocalBuilder(timeout = 10),
        runner = runner)
}

T = tuner(**option)

T.tune()








