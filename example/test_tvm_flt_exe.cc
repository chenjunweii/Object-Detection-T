
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

#include "src/tvm.hh"
#include "src/pipe.hh"
#include "src/executor.hh"
#include "src/detector.hh"
#include "src/box.hh"

#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

int main(){
 
	vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \ 
                                 "bus", "car", "cat", "chair", "cow", 
                               "diningtable", "dog", "horse", "motorbike", 
                               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };                                                          
	 
	string det_network = "ssd-inceptionv3-512-det";
	string nms_network = "ssd-inceptionv3-512-nms";
	
	string device = "gpu";

	map <string, vector <int64_t>> det_in;
	map <string, vector <int64_t>> det_out;
	
	vector <string> det_out_node = {"cls_prob", "loc_preds", "anchor_boxes"};
	
	map <string, vector <int64_t>> nms_out;

	det_in["data"] = {1, 3, 512, 512};
	det_out["cls_prob"] = {1, 21, 5186};
	det_out["loc_preds"] = {1, 20744};
	det_out["anchor_boxes"] = {1, 5186, 4};

	vector <string> nms_out_node = {"nms0_output"};

	nms_out["nms0_output"] = {1, 5186, 6};

	TVMExecutor det(det_network, det_in, det_out, det_out_node, device);

	Mat image = imread("dog.jpg");

	Mat resized;
	
	resize(image, resized, Size(512, 512));

	cvtColor(resized, resized, CV_BGR2RGB);

	string data = "data";

	det.Load(data, resized);

	det.Forward();

	det.GetOutput();

	TVMExecutor nms(nms_network, det_out, nms_out, nms_out_node, "cpu", false);

	nms.Load("cls_prob", det.nds["cls_prob"], false);

	nms.Load("loc_preds", det.nds["loc_preds"], false);
	
	nms.Load("anchor_boxes", det.nds["anchor_boxes"], false);

	nms.Forward();

	cout << "nms forward after" << endl;
	
	nms.GetOutput();

	for (int i = 0; i != 100; ++i)
	
		cout << nms.fs["nms0_output"][i] << ", ";

	vector <vector <bbox>> boxes;
	
	auto isize = image.size();

	convert(nms.fs["nms0_output"], boxes, isize);

	cout << "number of boxes : " << boxes[0].size() << endl;

	visualize(image, boxes, classes, 0.5);

	cout << endl;

	cout << "end" << endl;

    return 0;
}
