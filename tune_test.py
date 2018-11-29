from tune import *
import tvm
from tvm import autotvm
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'mxnet'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder = autotvm.LocalBuilder(timeout = 10),
        runner = autotvm.LocalRunner(number = 20, repeat = 3, timeout = 4),
    ),
}

tune_and_evaluate(tuning_option, network, target, dtype)

