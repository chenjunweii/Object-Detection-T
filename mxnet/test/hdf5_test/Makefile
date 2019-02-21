CC = g++


# Set the mxnet source directory

MXNET = ~/Source/mxnet
LOCAL_LIB = /usr/local/lib
MXNET_LIB = ${MXNET}/lib

LIBRARY = -L ${MXNET_LIB} \
-L ${LOCAL_LIB} \
-l boost_system \
-l boost_filesystem \
-l protobuf \
-l opencv_highgui \
-l opencv_imgcodecs \
-l opencv_imgproc \
-l opencv_core \
-l lmdb \
-l mxnet \
-l hdf5 \
-l hdf5_cpp


#CC_FILE = io.cc  kvstore.cc  ndarray.cc  operator.cc  optimizer.cc  symbol.cc




INCLUDE = -I ./ \
-I /usr/local/include \
-I ${MXNET}/include \
-I ${MXNET}/nnvm/include \
-I ${MXNET}/dmlc-core/include



ele: ElementwiseSum.cc

	${CC} ElementwiseSum.cc ${LIBRARY} ${INCLUDE} -o ele

hdf5_create_write: hdf5_create_write.cc

	${CC} hdf5_create_write.cc ${LIBRARY} ${INCLUDE} -o hdf5_create_write.out

mlp: mlp.cc

		${CC} mlp.cc ${LIBRARY} ${INCLUDE} -o mlp

read_keras: read_keras_pretrained.cc

		${CC} read_keras_pretrained.cc ${LIBRARY} ${INCLUDE} -o read_keras.out

char: charRNN.cc

		${CC} charRNN.cc ${LIBRARY} ${INCLUDE} -o char

header: test_header.cc

	${CC} test_header.cc ${LIBRARY} ${INCLUDE} -o header

not: broadcast_binary_not_operation.cc

	${CC} broadcast_binary_not_operation.cc ${LIBRARY} ${INCLUDE} -o not


concat: concat.cc

	${CC} concat.cc ${LIBRARY} ${INCLUDE} -o concat

read:read.cc

	${CC} read.cc ${LIBRARY} ${INCLUDE} -o read.out

create_group:create_group.cc

	${CC} create_group.cc ${LIBRARY} ${INCLUDE} -o create_group.out

infer: infer.cc

	${CC} infer.cc ${LIBRARY} ${INCLUDE} -o infer

infer_feed: infer_feed.cc

	${CC} infer_feed.cc ${LIBRARY} ${INCLUDE} -o infer_feed

feed_update: feed_update.cc

		${CC} feed_update.cc ${LIBRARY} ${INCLUDE} -o feed_update

broadcast_multiply: broadcast_multiply.cc

	${CC} broadcast_multiply.cc ${LIBRARY} ${INCLUDE} -o broadcast_multiply
