
#include "pipe.h"
#include <chrono>  // for high_resolution_clock
#include <numeric>
using namespace std;

TVMPipe::TVMPipe(vector <int64_t> & _shape, string device, string dtype = "float32", int _device_id = 0) : shape(_shape), device_id(_device_id) {

	size = accumulate(_shape.begin(), _shape.end(), 1, multiplies <int64_t> ());
	
	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;

	if (dtype.compare("float32") == 0){

		dtype_code = kDLFloat; dtype_bits = 32;
		
		dtype_lanes = 1; bytesize = size * 4;

	}

	else 

        throw invalid_argument("[!] Data Type is not support yet");
}

TVMPipe::TVMPipe(Mat & in, string device, string dtype = "float32", int device_id = 0, int batch = 1){

	c = in.channels();

	h = in.size().height;

	w = in.size().width;

	b = batch;

	ndim = 4;

<<<<<<< HEAD
=======
	//shape = new int64_t [ndim];
/*
	shape[0] = b;

	shape[1] = c;

	shape[2] = h;

	shape[3] = w;*/

>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;

	if (dtype.compare("float32") == 0){

		dtype_code = kDLFloat;
		
		dtype_bits = 32;
		
		dtype_lanes = 1;

		//size = b * c * w * h;

		//bytesize = size * 4;

	}

	else 

        throw invalid_argument("[!] Data Type is not support yet");

}


TVMPipe::~TVMPipe(){

	//delete shape;

}

void TVMPipe::FloatArrayToTVMArray(float * fdata, DLTensor * tensor){

<<<<<<< HEAD
	cout << "123124 " << endl;

	TVMArrayCopyFromBytes(tensor, fdata, bytesize);

}
void TVMPipe::MatToTVMArray(Mat & in, DLTensor * tensor, string & format){

	if (format.compare("NHWC") == 0){

		Mat f32;

		in.convertTo(f32, CV_32FC2);

		TVMArrayCopyFromBytes(tensor, f32.data, bytesize);

	}

	else if (format.compare("NCHW") == 0){

		vector <float> fdata (size);

		MatToFloatArray(in, fdata);
		
		TVMArrayCopyFromBytes(tensor, fdata.data(), bytesize);

	}

	else if (format.compare("darknet") == 0) {

		vector <float> fdata (size);

		//MatToFloatArrayYolo(in, fdata);

	}
=======
	Mat f32;

	in.convertTo(f32, CV_32FC2);

	TVMArrayCopyFromBytes(tensor, f32.data, bytesize);
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e

}
/*
void TVMPipe::MatToFloatArrayYolo(Mat & in, vector <float> & out){

	int cbase, hbase = 0;

	for (int _c = 0; _c < c; ++ _c) {

		cbase = _c * h * w;

		for (int _h = 0; _h < h; ++ _h) {

			hbase = h * _h;

			for (int _w = 0; _w < w; ++ _w) {

				out[cbase + hbase + _w] = (static_cast <float> (in.data[(hbase + _w ) * c + _c]) / 255.0);
			}
		}
	}
}
*/
void TVMPipe::MatToFloatArray(Mat & in, vector <float> & out){

	int cbase, hbase = 0;

	for (int _c = 0; _c < c; ++ _c) {

		cbase = _c * h * w;

		for (int _h = 0; _h < h; ++ _h) {

			hbase = h * _h;

			for (int _w = 0; _w < w; ++ _w) {

<<<<<<< HEAD
				out[cbase + hbase + _w] = (static_cast <int> (in.data[(hbase + _w ) * c + _c]));
				//out[cbase + hbase + _w] = (static_cast <float> (in.data[cbase + hbase + _w]));
=======
				//out[cbase + hbase + _w] = (static_cast <int> (in.data[(hbase + _w ) * c + _c]));
				out[cbase + hbase + _w] = (static_cast <float> (in.data[cbase + hbase + _w]));
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
			}
		}
	}
}

void TVMPipe::TVMArrayToFloatArray(DLTensor * in, vector <float> & out){

	TVMArrayCopyToBytes(in, out.data(), bytesize);
}

