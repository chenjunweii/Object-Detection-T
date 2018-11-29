




def load_mxnet_model(filename):

    out, arg_params, aux_params = mx.model.load_checkpoint('model/deploy_ssd_inceptionv3_512', 240) # must be deploy net

    out = mx.sym.load('model/deploy_ssd_inceptionv3_512-symbol.json')                                                                                         
    batch_shape = (1, 3, 512, 512)
     
    ctx = mx.gpu(1)            

