import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_device(0)

caffe.set_mode_gpu()

caffe.Net('ssd.prototxt', 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', caffe.TEST)

#caffe.Net('ssd2.prototxt', 'voc_coco.caffemodel', caffe.TEST)

# inception_b4_concat

# inception_c2_concat

