
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
#include <chrono>
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

	det_in["data"] = {1, 512, 512, 3};
	det_in["mean"] = {1, 3, 1, 1};
	det_out["cls_prob"] = {1, 21, 5186};
	det_out["loc_preds"] = {1, 20744};
	det_out["anchor_boxes"] = {1, 5186, 4};

	vector <string> nms_out_node = {"nms0_output"};

	nms_out["nms0_output"] = {1, 5186, 6};

	TVMExecutor det(det_network, det_in, det_out, det_out_node, device);
	
	TVMExecutor nms(nms_network, det_out, nms_out, nms_out_node, "cpu", false);

	auto start = chrono::high_resolution_clock::now();

	auto finish = chrono::high_resolution_clock::now();
	
	Mat image = imread("dog1.jpg");

	Mat resized;

	chrono::duration <double> total;
	chrono::duration <double> elapsed;

	for (int i = 0; i != 10; ++i){
	
		start = chrono::high_resolution_clock::now();
		
		resize(image, resized, Size(512, 512));
		
		finish = chrono::high_resolution_clock::now();

		elapsed = finish - start;

		total = elapsed;

		cout << "Reisize : " << elapsed.count() << endl;

		start = chrono::high_resolution_clock::now();
		cvtColor(resized, resized, CV_BGR2RGB);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "convert color : " << elapsed.count() << endl;
		
		string data = "data";

		start = chrono::high_resolution_clock::now();
		det.Load(data, resized);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "Load Image : " << elapsed.count() << endl;

		start = chrono::high_resolution_clock::now();
		det.Forward();
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "Det Forward : " << elapsed.count() << endl;
		
		start = chrono::high_resolution_clock::now();
		det.GetOutput(false);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "Det GetOutput : " << elapsed.count() << endl;
		
		start = chrono::high_resolution_clock::now();
		nms.Load("cls_prob", det.nds["cls_prob"], false);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "NMS Load Cls Prob : " << elapsed.count() << endl;

		start = chrono::high_resolution_clock::now();
		nms.Load("loc_preds", det.nds["loc_preds"], false);
		finish = chrono::high_resolution_clock::now();
		total += elapsed;
		elapsed = finish - start;
		cout << "NMS Load loc Pred: " << elapsed.count() << endl;


		start = chrono::high_resolution_clock::now();
		nms.Load("anchor_boxes", det.nds["anchor_boxes"], false);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "NMS Load Anchor : " << elapsed.count() << endl;

		start = chrono::high_resolution_clock::now();
		nms.Forward();
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "NMS Forward : " << elapsed.count() << endl;

		start = chrono::high_resolution_clock::now();
		nms.GetOutput(true);
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		total += elapsed;
		cout << "NMS GetOutput : " << elapsed.count() << endl;

		cout << "Total Elapsed time: " << total.count() << " s\n";


		cout << "------------------------" << endl;

	}

	vector <vector <bbox>> boxes;
	
	auto isize = image.size();

	convert(nms.fs["nms0_output"], boxes, isize);

	visualize(image, boxes, classes, 0.5);

    return 0;
}
