#ifndef FLT_MX_IMAGE_H
#define FLT_MX_IMAGE_H

#include <map>
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

#include "src/shape.h"

using namespace std;
using namespace mxnet::cpp;


namespace flt{

	namespace mx{

		namespace image{

			inline NDArray load(string filename, Context & ctx);
		
			inline void save(string filename, NDArray &nd, float scale = 1.0); // save image
			
			inline void save_1d(string filename, NDArray &nd, float scale = 1.0); // save image
			
			inline void saveb(string prefix, int number, vector <NDArray> &vnd); // save batch
			
			inline void saveb(string prefix, int number, NDArray & nd, float scale = 1.0);
			
			inline void saveb_1d(string prefix, int number, NDArray & nd, float scale = 1.0);

			inline void saveb(int number, vector <NDArray> & vnd);
			
			inline void saveb(vector <NDArray> &vnd);
			
			inline void load(string filename, NDArray & nd); // load image
			
			inline Symbol encode(Symbol s); // encode image
			
			inline Symbol encodeb(Symbol s); // encode batch

			inline Symbol decode(Symbol s); // decode image

			inline Symbol decodeb(Symbol s); // decode batch
			
			/*
			 
			   from a batch of Mat to single NDArray
			 
			*/

			inline void MatVector_to_NDArray(NDArray &n, vector <cv::Mat> &v, Context device);
			
			inline NDArray MatVector_to_NDArray(vector <cv::Mat> &v, Context device);
			
			inline vector <cv::Mat> NDArray_to_MatVector(NDArray &n);
			
			inline void Mat_to_FArray(cv::Mat & mat, vector <float> & farray);

			inline void Mat_to_NDArray(cv::Mat &mat, NDArray &nd);
			
			inline NDArray Mat_to_NDArray(cv::Mat &mat, Context &ctx);
			
			inline cv::Mat NDArray_to_Mat(NDArray &nd, float scale = 1.0);
			
			inline cv::Mat NDArray_to_Mat_1d(NDArray &nd, float scale = 1.0);

		} /* fimage */

	} /* fmx */

} /* flt */







#endif
