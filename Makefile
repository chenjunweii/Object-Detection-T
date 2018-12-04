CXX = g++
MXNET = ~/src/mxnet
TVM = ~/src/tvm
  
MXNET_LIB = ${MXNET}/lib

LIBRARY = -L ${MXNET_LIB} \
-l boost_system \
-l protobuf \
-l boost_filesystem \
-l opencv_highgui \
-l opencv_imgcodecs \
-l opencv_imgproc \
-l opencv_core \
-l opencv_video \
-l opencv_videoio \
-l mxnet \
-l pthread
  
INCLUDE = -I ./ \
-I /usr/local/include \
-I ${MXNET}/include \
-I ${TVM}/3rdparty/tvm/nnvm/include \
-I ${TVM}/3rdparty/dmlc-core/include \
-I src 
  
DEPENDENCIES = src/cv.h

# Makefile Example to deploy TVM modules.
#-TVM_ROOT=$(shell cd ../..; pwd)
TVM_ROOT = ~/src/tvm
NNVM_PATH = nnvm
DMLC_CORE = ${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I/opt/cuda/include

PKG_LDFLAGS = -L${TVM_ROOT}/build -L lib -ldl -lpthread -L/opt/cuda/lib64

.PHONY: clean all

all: lib/cpp_deploy_pack lib/cpp_deploy_normal


# Build rule for all in one TVM package library
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

test : test_tvm.cc lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -lcudart -lcublas -lcudnn -lcuda
test_flt : test_tvm_flt.cc lib/libtvm_runtime_pack.o src/*
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -lcudart -lcublas -lcudnn -lcuda $(LIBRARY)
test_flt_exe : test_tvm_flt_exe.cc lib/libtvm_runtime_pack.o src/*
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -lcudart -lcublas -lcudnn -lcuda $(LIBRARY)
# The code library built by TVM
lib/test_addone_sys.o: prepare_test_libs.py
	python prepare_test_libs.py

# Deploy using the all in one TVM package library
lib/cpp_deploy_pack: cpp_deploy.cc lib/test_addone_sys.o lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Deploy using pre-built libtvm_runtime.so
lib/cpp_deploy_normal: cpp_deploy.cc lib/test_addone_sys.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -ltvm_runtime


