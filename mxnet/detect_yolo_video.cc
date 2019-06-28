#include <map>
#include <thread>
#include <vector>
#include <iostream>
#include "src/image.hh"
#include "src/base.h"
#include "src/detector/yolo3.hh"

using namespace std;
using namespace flt::mx::image;
using namespace flt::mx;
using namespace mxnet::cpp;


int main(){

	string device = "gpu";

	string json = "yolo3-mobilenet1_0-symbol.json";
	
	string params = "yolo3-mobilenet1_0-0000.params";
	
	//string json = "yolo3-darknet53-symbol.json";
	
	//string params = "yolo3-darknet53-0000.params";

	/*vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \
								"bus", "car", "cat", "chair", "cow",
							  "diningtable", "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
							  */

	vector <string> classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

	string image = "dog.jpg";

	auto frame_size = cv::Size(1280, 720);

	bool switching = false;

	bool streaming = true;

	yolo3 det(json, params, device, classes, 224, frame_size, streaming, switching);
	
	DetectType dt = DetectType::video;
	
	for (int i = 0; i != 20; ++i){

		det.E[0]->Forward(false);
	}

	NDArray::WaitAll();

	string video = "Asakusa Street View Tokyo JAPAN-DFr-uP6iz40.mp4";
	//string video = "TimeSquare.mp4";
	
	bool multi_thread = true;

	thread det_1;

	if (multi_thread){

		thread capture (& yolo3::capture, & det, ref(dt), ref(video));

		int tid_0 = 0; int tid_1 = 1;

		thread det_0 (& yolo3::detect, & det, ref(tid_0));

		if (switching)
		
			det_1 = thread(& yolo3::detect, & det, ref(tid_1));

		det.post(true);
		
		capture.join();

		det_0.join();

		if (switching)
		
			det_1.join();
	
	}

	else
	
		det.detect_video(video);
    
	//
	


	//det.detect_image(img);
	//

	

	MXNotifyShutdown();

	return 0;

}

