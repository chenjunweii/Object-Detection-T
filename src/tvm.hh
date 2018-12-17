

#include "tvm.h"
#include <stdexcept>

using namespace std;
using namespace cv;



void MatToFloatArray(Mat & mat, vector <float> & farray){

	int h = mat.size().height;

	int w = mat.size().width;

	int channel = mat.channels();

	int cbase, hbase = 0;

	for (int c = 0; c < channel; ++c) {

		cbase = c * h * w;

		for (int i = 0; i < h; ++i) {

			hbase = h * i;

			for (int j = 0; j < w; ++j) {

					farray[cbase + hbase + j] = (static_cast <int> (mat.data[(hbase + j) * channel + c]));
			}
		}
		
	}
}

void MatToTVMArray(Mat & m, DLTensor * x, string device, string dtype = "float32", bool allocate = true, int device_id = 0, int ndim = 4){

	int c = m.channels();

	auto cvsize = m.size();

	int w = cvsize.width;

	int h = cvsize.height;

	int size = c * h * w;

	int bytesize;
	
	int64_t shape[ndim] = {1, c, h, w};

	int dtype_code, dtype_bits, dtype_lanes, device_type;

	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;
    

	if (dtype.compare("float32") == 0){

		bytesize = size * 4; // 4 stands for float32, 4 bytes
		
		dtype_code = kDLFloat;

		dtype_bits = 32;
		
		dtype_lanes = 1;
		
	}

	else {

        throw invalid_argument("[!] Data Type is not support yet");
	}


	vector <float> farray (size);

	MatToFloatArray(m, farray);

	if (allocate)

		TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

	TVMArrayCopyFromBytes(x, farray.data(), bytesize);

}


void TVMArrayToFloatArray(DLTensor * x, vector <float> & farray){

	// Not Yet

}



