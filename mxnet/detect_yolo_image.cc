#include <iostream>
#include <map>
#include <thread>
#include <vector>
#include "src/image.hh"
#include "src/base.h"
#include "src/detector/yolo3.hh"

using namespace std;
using namespace flt::mx::image;
using namespace flt::mx;
using namespace mxnet::cpp;


int main(){

	string device = "gpu";
	
	string json = "yolo3-darknet53-symbol.json";
	
	string params = "yolo3-darknet53-0000.params";

	/*vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \
								"bus", "car", "cat", "chair", "cow",
							  "diningtable", "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
							  */

	vector <string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

	string image = "dog.jpg";

	auto img = cv::imread(image);

	auto frame_size = img.size();

	yolo3 det(json, params, device, classes, 224, frame_size, false);

	det.detect_image(img);

	MXNotifyShutdown();

	return 0;

}

