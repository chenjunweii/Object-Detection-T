
#include "pipe.h"
using namespace std;


TVMPipe::TVMPipe(vector <int64_t> & _shape, string device, string dtype = "float32", int _device_id = 0) : shape(_shape), device_id(_device_id) {

	ndim = shape.size();

	if (ndim == 3){

		b = 1;

		c = shape[0];

		h = shape[1];

		w = shape[2];
	}

	else if (ndim == 4){

		b = shape[0];

		c = shape[1];

		h = shape[2];

		w = shape[3];

	}

	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;

	if (dtype.compare("float32") == 0){

		dtype_code = kDLFloat;
		
		dtype_bits = 32;
		
		dtype_lanes = 1;

		size = b * c * w * h;

		bytesize = size * 4;

	}

	else 

        throw invalid_argument("[!] Data Type is not support yet");

	cout << "hhh" << endl;
}

TVMPipe::TVMPipe(Mat & in, string device, string dtype = "float32", int device_id = 0, int batch = 1){

	c = in.channels();

	h = in.size().height;

	w = in.size().width;

	b = batch;

	ndim = 4;

	//shape = new int64_t [ndim];

	shape[0] = b;

	shape[1] = c;

	shape[2] = h;

	shape[3] = w;

	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;

	if (dtype.compare("float32") == 0){

		dtype_code = kDLFloat;
		
		dtype_bits = 32;
		
		dtype_lanes = 1;

		size = b * c * w * h;

		bytesize = size * 4;

	}

	else 

        throw invalid_argument("[!] Data Type is not support yet");

}


TVMPipe::~TVMPipe(){

	//delete shape;

}

void TVMPipe::MatToTVMArray(Mat & in, DLTensor * tensor){

	vector <float> farray (size);
	
	TVMPipe::MatToFloatArray(in, farray);

	//if (tensor == nullptr)

	//	TVMArrayAlloc(shape.data(), ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, & tensor);

	TVMArrayCopyFromBytes(tensor, farray.data(), bytesize);

}


void TVMPipe::MatToFloatArray(Mat & in, vector <float> & out){

	int cbase, hbase = 0;

	for (int _c = 0; _c < c; ++ _c) {

		cbase = _c * h * w;

		for (int _h = 0; _h < h; ++ _h) {

			hbase = h * _h;

			for (int _w = 0; _w < w; ++ _w) {

				out[cbase + hbase + _w] = (static_cast <int> (in.data[(hbase + _w ) * c + _c]));
			}
		}
	}
}

void TVMPipe::TVMArrayToFloatArray(DLTensor * in, vector <float> & out){

	TVMArrayCopyToBytes(in, out.data(), bytesize);
}

