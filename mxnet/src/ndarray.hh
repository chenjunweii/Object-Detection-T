#ifndef FLT_MX_NDARRAY_HH
#define FLT_MX_NDARRAY_HH

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include "ndarray.h"

using namespace std;
using namespace mxnet::cpp;



inline NDArray flt::mx::nd::FArray_to_NDArray(float * farray, Shape shape, Context device){
	
	NDArray ndtotal(shape, device);
	
	//cout << "after total " << endl;

	//cout << shape.Size() << endl;

	ndtotal.SyncCopyFromCPU(farray, shape.Size());

	//cout << "AFte sync" << endl;
	NDArray::WaitAll();

	return ndtotal;

}


inline void flt::mx::nd::FArray_to_NDArray(float * farray, NDArray * nd, Shape shape, Context device){
	
	(*nd).SyncCopyFromCPU(farray, shape.Size());

	NDArray::WaitAll();
}

inline void flt::mx::nd::NDArray_to_FArray(NDArray * nd, float * array, Shape shape){

	(*nd).SyncCopyToCPU(array, shape.Size());

}

#endif
