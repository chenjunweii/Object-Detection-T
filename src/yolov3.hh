#ifndef DETECTOR_YOLOV3_DARKNET_HH
#define DETECTOR_YOLOV3_DARKNET_HH

#include "yolov3.h"
#include "darknet/image.hh"
#include "darknet/yolo.hh"

using namespace flt::detector;
using namespace flt::darknet;

yolov3::yolov3(string _network, string _device, Size _size, vector <string> & _classes,
		map <string, vector <int64_t>> & _det_in, map <string, vector <int64_t>> & _det_out) : 
			network(_network), device(_device), classes(_classes),
			det_in(_det_in), det_out(_det_out), size(_size) {


	string det_network = _network;

	cout << "create executor" << endl;
	
	det = new TVMExecutor(det_network, _det_in, _det_out, det_out_node, _device);

}


void yolov3::letterbox(Mat & in){

}


void yolov3::preprocess(Mat & in){

	//convertTo(in, in, 1);

    //image sized = letterbox_image(im, size.width, size.height);
}

void show10(float * f, string comment){

	cout << comment << " : [";

	for (int i = 0; i != 10; ++i)

		cout << f[i + 50000] << ", ";

	cout << "]" << endl;
}
void yolov3::detect_image(string sin) {

	Mat in = imread(sin);

	flt::darknet::image yin = load_image(in, 0, 0, 3);

	flt::darknet::image ysized = letterbox_image(yin, size.width, size.height);

	show10(yin.data, "yin");
	
	show10(ysized.data, "ysized");

	det->Load("data", ysized.data);

	det->Forward();

	det->GetOutput(true);

	//find_active((float*) det->nds["reshape5_output"]->data, 0.5);
	//find_active((float*) det->fs["reshape5_output"].data(), 0.5);
	
	vector <vector <float>> layer_out;

	layer_out.push_back(det->fs["reshape5_output"]);
	layer_out.push_back(det->fs["reshape3_output"]);
	layer_out.push_back(det->fs["reshape1_output"]);

	vector <int> lw = {52, 26, 13};
	vector <int> lh = {52, 26, 13};

	int nboxes = 0;

	//detection * dets = get_detections(layer_out, lw, lh, osize.width, osize.height, &nboxes);
	//
	
	vector <layer> layers;

	layer l("reshape5", det->fs, det_out);

	int hh;

	detection xx;

	get_yolo_detections(l, osize.width, osize.height, 416, 416, 0.5, & hh, 1, & xx);
	//int            get_yolo_detections(layer l,int w, int h, int netw,int neth, float thresh, int * map,int relative, detection * dets) {

	//convert((float *)nms->nds["nms0_output"]->data, boxes, isize);

	//visualize(in, boxes, 0.5, 0);

}


inline void yolov3::find_active(float * flo, float threshold){

	vector <vector <int>> location;

	for (int i = 0; i != 3; ++i){

		location.emplace_back(vector <int> ());

		int ibase = i * 85 * 52 * 52;

		int d = 4; // for

		int dbase = d * 52 * 52;

		for (int hw = 0; hw != 52 * 52; ++hw)

			if (flo[ibase + dbase + hw] > threshold)

				location[i].emplace_back(ibase + dbase + hw);

			printf("%d Find %d Active\n", i, location[i].size());
		
	}

}


inline void yolov3::get_box(){

}

inline void yolov3::nms(){

}

inline yolov3::~yolov3(){
	
	delete det;
};



inline void yolov3::convert(float * fout, vector <vector <bbox>> & bboxes, Size & size_){

	
	for(int l = 0; l != 3; ++l){


	}

	/*

	vector <float> slice;
	for (int i = 0; i != 1; ++i){
			(bboxes).emplace_back(vector <bbox> ());
		for (int j = 0; j != int(nms_out["nms0_output"][1]); ++j){
			slice = vector <float> (fout + j * 6, fout + j * 6 + 6);
			if (slice[0] >= 0){
				(bboxes)[i].emplace_back(move(bbox(slice, size_)));
			}
		}
	}*/
}


inline void yolov3::visualize(Mat & in, vector <vector <bbox>> & bboxes, float threshold, int stay){
		assert(bboxes.size() == 1); // assert batch size = 1
		for (auto & box : bboxes[0]){
			if (box.s >= threshold){
				Point ul(box.x, box.y);
				Point br(box.x1, box.y1);
				Point tp(box.x1, box.y1 - 5);
				rectangle(in, ul, br, cv::Scalar(0, 255, 0), 3.5);
				string text = boost::str(boost::format("%s : %f") % classes[box.c] % box.s);
				putText(in, text, tp, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,  cv::Scalar(0, 0, 255, 255), 2);
				cout << "classes : " << classes[box.c] << endl;
			}
		}

		cv::imshow("Visualize", in);

		cv::waitKey(stay);
}





#endif
