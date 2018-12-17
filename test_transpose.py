from mxnet import nd
import mxnet as mx
from time import time

cpu = mx.cpu(0)
gpu = mx.gpu(0)

mean_cpu = nd.zeros([1, 3, 1, 1], cpu)

mean_gpu = nd.zeros([1, 3, 1, 1], gpu)

image_cpu = nd.zeros([1, 512, 512, 3], cpu)

image_gpu = nd.zeros([1, 512, 512, 3], gpu)

start = time()

swap_cpu = image_cpu.transpose([0, 3, 1, 2]) - mean_cpu

print(swap_cpu.shape)

print('CPU : ', time() - start)

start = time()

swap_gpu = image_gpu.transpose([0, 3, 1, 2]) - mean_gpu

print('GPU : ', time() - start)
