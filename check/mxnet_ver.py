import os
import time
import mxnet as mx
import numpy as np

out, arg_params, aux_params = mx.model.load_checkpoint('deploy_ssd_mobilenet_v2_300', 1) # must be deploy net

shape = (1, 3, 300, 300)

ctx = mx.gpu(0)

executor = out.simple_bind(ctx = ctx,
                           data = shape,
                           grad_req = 'null',
                           force_rebind = True)

executor.copy_params_from(arg_params, aux_params)

for i in range(10):
    
    data = mx.nd.array(np.zeros(shape))

    start = time.time()

    y_gen = executor.forward(is_train = False, data = data)

    y_gen[0].asnumpy()

    print('Time : ', time.time() - start)
