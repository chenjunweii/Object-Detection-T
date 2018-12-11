#include <iostream>
#include "flt.h"
#include <map>
#include <thread>
#include <vector>
#include "detector.hh"


using namespace std;
using namespace flt::mx::image;
using namespace flt::mx;
using namespace mxnet::cpp;

int main(){

	string device = "gpu";
	
	Size size (512, 512);
	
	string json = "deploy_ssd_inceptionv3_512-symbol.json";
	
	string params = "deploy_ssd_inceptionv3_512-0215.params";

	string mean = "mean.nd";

	vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \
								"bus", "car", "cat", "chair", "cow",
							  "diningtable", "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	detector det(json, params, mean, device, classes, size, false);

	DetectType dt = DetectType::video;

	//string video = "av10239720.mp4";
	//
	//string video = "Asakusa Street View Tokyo JAPAN-DFr-uP6iz40.mp4";
	string video = "av27133366.mp4";

	thread capture (& detector::capture, & det, ref(dt), ref(video));

	int tid_0 = 0; int tid_1 = 1;

    thread det_0 (& detector::detect, & det, ref(tid_0));
    
	//thread det_1 (& detector::detect, & det, ref(tid_1));

	det.post(true);
    
	capture.join();

    det_0.join();
    
	//det_1.join();

	MXNotifyShutdown();

	return 0;

}

