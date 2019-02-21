#ifndef FLT_MX_NDARRAY_H
#define FLT_MX_NDARRAY_H

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace mxnet::cpp;

namespace flt{
	
	namespace mx{
		
		namespace nd{

			inline NDArray FArray_to_NDArray(float * farray, Shape shape, Context device);
			inline void FArray_to_NDArray(float * farray, NDArray * nd, Shape shape, Context device);
			inline void NDArray_to_FArray(NDArray * nd, float * farray, Shape shape);
		
		}/* nd */

	} /* fmx */

} /* flt */

#endif
