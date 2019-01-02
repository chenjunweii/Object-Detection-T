import caffe

net = caffe.Net('inceptionv3.prototxt', None, caffe.TEST)

"""
print("Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

"""

"""

print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

"""
