#ifndef FLT_PIPE_H
#define FLT_PIPE_H

#include "tvm.h"

class TVMPipe {

	public:

		int b, c, w, h;

		int size, bytesize;

		bool image;

		string format;

		vector <int64_t> shape;

		int dtype_code, dtype_bits, dtype_lanes, device_type, device_id, ndim;

		inline TVMPipe(vector <int64_t> & shape, string device, string dtype, int device_id);

		inline TVMPipe(Mat & in, string device, string dtype, int device_id, int batch);

		inline TVMPipe(){};

		inline ~ TVMPipe();

		inline void MatToTVMArray(Mat & in, DLTensor * out, string & format);

		inline void TVMArrayToFloatArray(DLTensor * in, vector <float> & out);

		inline void FloatArrayToTVMArray(float * fdata, DLTensor * out);

	private:

		inline void MatToFloatArray(Mat & in, vector <float> & out);
	
		inline void FloatArrayToMat(vector <float> & in, Mat & out);

};



#endif
