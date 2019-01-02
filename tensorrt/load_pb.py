import tensorflow as tf
import numpy as np

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    #with tf.gfile.GFile('mobilenetv2.pb', 'rb') as fid:
    #pb = 'inception_v2.pb'
    pb = 'mobilenetv2.pb'
    with tf.gfile.GFile(pb, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
    with tf.Session(graph=detection_graph) as sess:
        #"""
        for n in detection_graph.as_graph_def().node:
            if not 'Post' in n.name:
                print(n.name)
        """

        for op in detection_graph.get_operations():
            if not 'Post' in op.name:
                print(str(op.name))

        """

        image_np = np.zeros((300, 300, 3))
        
        image_np_expanded = np.expand_dims(image_np, axis = 0)
        
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        concat = detection_graph.get_tensor_by_name('concat:0')

        concat_reshape = tf.reshape(concat, [1, 1917, -1], name = 'concat')
        
        concat = detection_graph.get_tensor_by_name('concat:0')
        
        concat_1 = detection_graph.get_tensor_by_name('concat_1:0')

        concat, concat_1 = sess.run([concat, concat_1], feed_dict = {image_tensor: image_np_expanded})

        print('pb : ', pb)

        print('concat : ', concat.shape)
        
        print('concat 1 : ', concat_1.shape)
        
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

