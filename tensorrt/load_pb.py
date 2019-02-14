import tensorflow as tf
import numpy as np

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    #with tf.gfile.GFile('mobilenetv2.pb', 'rb') as fid:
    pb = 'inception_v2_512.pb'
    #pb = 'inception_v3_voc.pb'
    with tf.gfile.GFile(pb, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
    with tf.Session(graph=detection_graph) as sess:
        #"""
        for n in detection_graph.as_graph_def().node:
            if 'Detection' in n.name:
                print(n.name)
        """

        for op in detection_graph.get_operations():
            if not 'Post' in op.name:
                print(str(op.name))

        """

        image_np = np.zeros((512, 512, 3))
        
        image_np_expanded = np.expand_dims(image_np, axis = 0)
        
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        concat = detection_graph.get_tensor_by_name('Concatenate/concat:0')

        ndet = 6

        boxes = [None] * 6

        symbols = [None] * 6

        for i in range(ndet):
        
            symbols[i] = detection_graph.get_tensor_by_name('BoxPredictor_{}/BoxEncodingPredictor/Conv2D:0'.format(i))

        #concat_1 = detection_graph.get_tensor_by_name('FeatureExtractor/MobilenetV2/expanded_conv_13/expansion_output:0')

        boxes = sess.run(symbols, feed_dict = {image_tensor: image_np_expanded})

        print('pb : ', pb)

        for box in boxes:

            print(box.shape)

        raise

        for op in detection_graph.get_operations():

            if not 'Post' in op.name:
                
                print(str(op.name))

        """
        
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        scores = detection_graph.get_tensor_by_name('detection_scores:0')

        classes = detection_graph.get_tensor_by_name('detection_classes:0')

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict = {image_tensor: image_np_expanded})

        print(boxes.shape)

        print(scores.shape)

        print(classes.shape)

        print(num_detections.shape)
        """

