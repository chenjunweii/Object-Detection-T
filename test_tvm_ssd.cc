

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
#include "src/ssd.hh"

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
	 
	string network = "ssd-inceptionv3-512";
	
	string device = "gpu";

	map <string, vector <int64_t>> det_in, det_out, nms_out;

	det_in["data"] = {1, 512, 512, 3};
	det_in["mean"] = {1, 3, 1, 1};
	det_out["cls_prob"] = {1, 21, 5186};
	det_out["loc_preds"] = {1, 20744};
	det_out["anchor_boxes"] = {1, 5186, 4};
	nms_out["nms0_output"] = {1, 5186, 6};
	
	Size size(512, 512);

	ssd s(network, device, size, classes, det_in, det_out, nms_out);

	//s.detect_image("dog1.jpg");
	//
	
<<<<<<< HEAD
=======
	
>>>>>>> 3df6457f817f3ee5923f83d0c9377e0a1a19fc2e
	string video = "123.mp4";
	
	DetectType dt = DetectType::video;

	thread capture (& ssd::capture_thread, & s, ref(dt), ref(video));

    thread detect (& ssd::detect_thread, & s);

	s.post_thread(true);
    
	capture.join();

    detect.join();

}
