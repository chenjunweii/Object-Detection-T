SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
LOCALLIB?="~/trt/lib"
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
TRIPLE=x86_64-linux
ifeq ($(TARGET), aarch64)
TRIPLE=aarch64-linux
endif
ifeq ($(TARGET), qnx)
TRIPLE=qnx-linux
endif
ifeq ($(TARGET), android32)
TRIPLE=arm-linux-androideabi
endif
ifeq ($(TARGET), android64)
TRIPLE=aarch64-linux-android
endif
export TARGET
export VERBOSE
export LIBDIR
export TRIPLE
export LOCALLIB
samples=sampleUffSSD.cpp
.PHONY: all clean help test test_debug test_release

LIBRARY = -l boost_system \
-l protobuf \
-l boost_filesystem \
-l opencv_highgui \
-l opencv_imgcodecs \
-l opencv_imgproc \
-l opencv_core \
-l opencv_video \
-l opencv_videoio \
-l pthread

ssd:

	g++ sampleUffSSD.cpp -L ~/trt/lib -I /opt/cuda-9/include -I /opt/tensorrt-4/include -L /opt/cuda-9/lib64 -lcudnn -lcudart -lcublas -lnvinfer_plugin -lnvinfer -lnvparsers

serialize: serialize.cc
	
	g++ serialize.cc -L ~/trt/lib -I . -I /opt/cuda-9/include -I /opt/tensorrt-4/include -L /opt/cuda-9/lib64 -lcudnn -lcudart -lcublas -lnvinfer_plugin -lnvinfer -lnvparsers -o serialize $(LIBRARY)

detect: test_detect.cc src/*
	
	g++-7.3.1 -std=c++14 test_detect.cc -L ~/trt/lib -I . -I /opt/cuda-9/include -I /opt/tensorrt-4/include -L /opt/cuda-9/lib64 -lcudnn -lcudart -lcublas -lnvinfer_plugin -lnvinfer -lnvparsers -o detect $(LIBRARY) -O3

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

test: test_debug test_release

test_debug:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_debug -C $(sample) &&) :

test_release:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_release -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\ttest[_debug|_release] - test all/release/debug samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."

