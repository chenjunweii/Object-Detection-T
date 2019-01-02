import os
import tvm
import nnvm
import numpy as np
from nnvm import testing
from tvm import autotvm
from tvm.contrib.util import tempdir
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import tvm.contrib.graph_runtime as runtime

from build import get_network
from utils import save_tvm_graph, save_tvm_params

class tuner(object):
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.net, self.params, self.input_shape, self.out_shape = get_network(self.network, batch_size = 1, input_size = self.input_size)
    
    def tune(self):
        print("[*] Extract tasks...")
        print('Target Host : ', self.target_host)
        #"""
        self.tasks = autotvm.task.extract_from_graph(self.net, target = self.target, target_host = self.target_host,
                shape = {'data': self.input_shape, 'mean' : (1, 3, 1, 1)}, dtype = self.dtype,
                                                symbols = (nnvm.sym.conv2d, nnvm.sym.dense))
                                                    #nnvm.symbol.multibox_prior,
                                                    #nnvm.symbol.multibox_transform_loc,
                                                    #nnvm.symbol.nms))
        #"""


        # run tuning tasks
        print("[*] Tuning...")

        #if not self.recompile:
        self.tune_tasks()

        print("[*] Compile...")
        #with autotvm.apply_history_best(self.log_filename):
        with nnvm.compiler.build_config(opt_level = 3):
            graph, lib, params = nnvm.compiler.build(
                self.net,
                self.target,
                target_host = self.target_host,
                shape = {'data': self.input_shape, 'mean' : (1, 3, 1, 1)},
                params = self.params,
                dtype = self.dtype)

        # export library

            print('[*] Exporting ... ')
            
            lib.export_library('lib/{}.tvm.so'.format(self.network))

            lib.save('lib/{}.tvm.o'.format(self.network))

            save_tvm_graph(self.network, graph)
    
    def tune_tasks(self,
                    use_transfer_learning = True,
                    try_winograd = True):
        if try_winograd:
            for i in range(len(self.tasks)):
                try:  # try winograd template
                    tsk = autotvm.task.create(self.tasks[i].name, self.tasks[i].args,
                                              self.tasks[i].target, self.tasks[i].target_host, 'winograd')
                    input_channel = tsk.workload[1][1]
                    if input_channel >= 64:
                        tasks[i] = tsk
                except Exception:
                    pass

        # create tmp log file
        tmp_log_file = self.log_filename + ".tmp"
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(self.tasks)):
            
            prefix = "[Task %2d/%2d] " %(i+1, len(self.tasks))

            # create tuner
            if self.tuner == 'xgb' or self.tuner == 'xgb-rank':
                tuner_obj = XGBTuner(tsk, loss_type = 'rank')
            elif self.tuner == 'ga':
                tuner_obj = GATuner(tsk, pop_size=100)
            elif self.tuner == 'random':
                tuner_obj = RandomTuner(tsk)
            elif self.tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + self.tuner)

            if use_transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

            # do tuning
            tuner_obj.tune(n_trial = min(self.n_trial, len(tsk.config_space)),
                           early_stopping = self.early_stopping,
                           measure_option = self.measure_option,
                           callbacks = [
                               autotvm.callback.progress_bar(self.n_trial, prefix = prefix),
                               autotvm.callback.log_to_file(tmp_log_file)])

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, self.log_filename)
        os.remove(tmp_log_file)

    def evaluate(self, target, graph, lib, input_shape):

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
