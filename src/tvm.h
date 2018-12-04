#ifndef FLT_TVM_H
#define FLT_TVM_H

#include <vector>
#include <iostream>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace tvm::runtime;

inline void MatToFloatArray(cv::Mat & mat, vector <float> & farray);

inline void MatToTVMArray(Mat & m, DLTensor * x, string device, string dtype, bool allocat, int device_id, int ndim);

inline void TVMArrayToFloatArray(DLTensor * x, vector <float> & farray);


#endif
