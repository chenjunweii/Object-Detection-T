

def get_coco(target):
    block = model_zoo.yolo3_darknet53_coco(pretrained_base=True, pretrained=True)
    net, params = relay.frontend.from_mxnet(block, {"data": dshape})
    #with relay.build_config(opt_level=3):
    #graph, lib, params = relay.build(mod[mod.entry_func], target, params=params)
 
    #print(target)
    return net, params


