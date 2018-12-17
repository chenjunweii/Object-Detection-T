

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <thread>
#include <algorithm>

#include "src/pipe.hh"
#include "src/executor.hh"
#include "src/detector.hh"
#include "src/box.hh"
#include "src/yolov3.hh"

#include <opencv2/opencv.hpp> 
#include <chrono>

using namespace std;
using namespace cv;
using namespace flt::detector;

int main(){

	vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \ 
                                 "bus", "car", "cat", "chair", "cow", 
                               "diningtable", "dog", "horse", "motorbike", 
                               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };                                                          
	 
	string network = "yolov3-darknet-416";
	
	string device = "gpu";

	map <string, vector <int64_t>> det_in, det_out;

	det_in["data"] = {3, 416, 416};
	//det_in["mean"] = {1, 3, 1, 1};
	det_out["reshape5_output"] = {1, 255, 52, 52};
	det_out["reshape5_mask"] = {3};
	det_out["reshape5_bias"] = {18};
	det_out["reshape5_attr"] = {6};
	det_out["reshape3_output"] = {1, 255, 26, 26};
	det_out["reshape3_mask"] = {3};
	det_out["reshape3_bias"] = {18};
	det_out["reshape3_attr"] = {6};
	det_out["reshape1_output"] = {1, 255, 13, 13};
	det_out["reshape1_mask"] = {3};
	det_out["reshape1_bias"] = {18};
	det_out["reshape1_attr"] = {6};

	int nclasses = 80;
	//outshape	(3, 85, 52, 52) // 255 / 3 = 85
	//out shape :  (3, 85, 26, 26)
	//out shape :  (3, 85, 13, 13)

	Size size(416, 416);

	yolov3 y(network, device, size, classes, det_in, det_out);

	y.detect_image("dog.jpg");
	//
	/*
	string video = "123.mp4";
	
	DetectType dt = DetectType::video;

	thread capture (& ssd::capture_thread, & s, ref(dt), ref(video));

    thread detect (& ssd::detect_thread, & s);

	s.post_thread(true);
    
	capture.join();

    detect.join();

	*/

}
