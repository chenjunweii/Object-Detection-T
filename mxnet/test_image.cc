
#include <iostream>
#include "flt.h"
#include <map>
#include <thread>
#include <vector>
#include "detector.hh"
#include "image.hh"

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

	string image = "dog1.jpg";

	det.detect_image(image);

	MXNotifyShutdown();

	return 0;

}

