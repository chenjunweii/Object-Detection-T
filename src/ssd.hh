#ifndef DETECTOR_SSD_HH
#define DETECTOR_SSD_HH

#include "ssd.h"
#include "detector.hh"

/* TVM Version */

using namespace flt::detector;



ssd::ssd(string _network, string _device, Size _size, vector <string> & _classes,
		map <string, vector <int64_t>> & _det_in, map <string, vector <int64_t>> & _det_out, 
		map <string, vector <int64_t>> & _nms_out) : network(_network), device(_device), classes(_classes),
			det_in(_det_in), det_out(_det_out), nms_out(_nms_out), size(_size) {

	string det_network = _network + "-det";
	
	string nms_network = _network + "-nms";
	
	det = new TVMExecutor(det_network, _det_in, _det_out, det_out_node, _device);
	
	nms = new TVMExecutor(nms_network, _det_out, _nms_out, nms_out_node, "cpu", false);
}


inline int ssd::capture_thread(DetectType & dt, string & filename){

	cout << "in Capture Thread " << endl;
	
	Mat frame, resized;
	VideoCapture capture;
	
	if (dt == DetectType::video){
		capture = VideoCapture(filename);

		cout << "is video" << endl;

	}
	
	if (!capture.isOpened()){

		cout << "not open" << endl;
		return -1;
	}

	detect_type = dt;

	osize = Size(int(capture.get(CV_CAP_PROP_FRAME_WIDTH)), int(capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

	bool get = false;
	
	alive["capture"] = true;

	int idx = 0;
    
	while(true){

		cout << "Capture : " << idx << endl;

		idx += 1;
		while(OriginalQueue.size() > 15){
			cout << "capture > 15 " << endl;
			usleep(wait * 3);

		}
		capture >> frame;
		if(frame.empty())
			break;

		resize(frame, resized, Size(size.width, size.height));
		cvtColor(resized, resized, CV_BGR2RGB);
		
		get = false;
		
		while (! get){
			get = mOriginalQueue.try_lock();
			if (not get)
				usleep(wait);
		}
		OriginalQueue.emplace_back(move(frame));
		mOriginalQueue.unlock();

		get = false;
		while (! get){
			get = mResizedQueue.try_lock();
			if (not get)
				usleep(wait);
		}

		ReSizedQueue.emplace_back(move(resized));
		mResizedQueue.unlock();

		usleep(wait);
		if (not alive["capture"])
			break;
    }
	alive["capture"] = false;
    return 0;	

}


void ssd::detect_thread(){

	bool get = false;

	Mat resized;

	while (true){

		get = false;

		while (! get){
			cout << "[!] Detect Acquire ReSizedQueue " << endl;
			while (ReSizedQueue.empty()){

				cout << "[!] Detect Acquire ResizedQueue, But Still Empty " << endl;

				usleep(wait);
			}
			get = mResizedQueue.try_lock();
		}

		resized = move(ReSizedQueue.front());
		ReSizedQueue.pop_front();
		ReSizedQueue.shrink_to_fit();
		mResizedQueue.unlock();
		
		det->Load("data", resized, "NHWC");
		det->Forward();
		det->GetOutput(false);
		
		nms->Load("cls_prob", det->nds["cls_prob"], false);
		nms->Load("loc_preds", det->nds["loc_preds"], false);
		nms->Load("anchor_boxes", det->nds["anchor_boxes"], false);
		
		nms->Forward();
		nms->GetOutput(false);

		vector <vector <bbox>> boxes;

		convert((float*)nms->nds["nms0_output"]->data, boxes, osize);

		get = false;

		while (! get){

			cout << "Detect Aquiring BoxQueue" << endl;
			get = mBoxesQueue.try_lock();
			if (not get)
				usleep(wait);
		}

		BoxesQueue.emplace_back(move(boxes));
		mBoxesQueue.unlock();
	
	}

}

inline ssd::~ssd(){

	delete det;

	delete nms;

}

inline int ssd::guard(){
	for (auto & i : alive){
		if (i.second == false){
			cout << "[*] Guard Close " << i.first << endl;
			for (auto & j : alive){
				j.second = false;
			}
		}
	}
}

void ssd::detect_image(string sin){

	Mat resized;
	Mat in = imread(sin);
	resize(in, resized, Size(size.width, size.height));
	cvtColor(resized, resized, CV_BGR2RGB);

	det->Load("data", resized, "NHWC");
	det->Forward();
	det->GetOutput(false);
	
	nms->Load("cls_prob", det->nds["cls_prob"], false);
	nms->Load("loc_preds", det->nds["loc_preds"], false);
	nms->Load("anchor_boxes", det->nds["anchor_boxes"], false);
	
	nms->Forward();
	nms->GetOutput(false);
	
	vector <vector <bbox>> boxes;
	
	auto isize = in.size();

	convert((float *)nms->nds["nms0_output"]->data, boxes, isize);

	visualize(in, boxes, 0.5, 0);

}


inline void ssd::convert(float * fout, vector <vector <bbox>> & bboxes, Size & size_){

	vector <float> slice;
	for (int i = 0; i != 1; ++i){
			(bboxes).emplace_back(vector <bbox> ());
		for (int j = 0; j != int(nms_out["nms0_output"][1]); ++j){
			slice = vector <float> (fout + j * 6, fout + j * 6 + 6);
			if (slice[0] >= 0){
				(bboxes)[i].emplace_back(move(bbox(slice, size_)));
			}
		}
	}
}


inline void ssd::visualize(Mat & in, vector <vector <bbox>> & bboxes, float threshold, int stay){
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

void ssd::post_thread(bool viz){

	bool get = false;
	alive["capture"] = true;
	cv::Mat frame;
	while (true){

		while (! get){
			while (BoxesQueue.empty()){
				usleep(wait);
			}
			get = mBoxesQueue.try_lock();
		}

		vector <vector <flt::bbox>> boxes = move(BoxesQueue.front());

		BoxesQueue.pop_front();
		BoxesQueue.shrink_to_fit();
		mBoxesQueue.unlock();
		get = false;

		while (! get){
			get = mOriginalQueue.try_lock();
			usleep(wait);
		}

		get = false;
		frame = move(OriginalQueue.front());
		OriginalQueue.pop_front();
		OriginalQueue.shrink_to_fit();
		mOriginalQueue.unlock();

		if (viz)
			visualize(frame, boxes, 0.5, 33);
		guard();
		usleep(wait);
		if (not alive["capture"])
			break;

	}
}



#endif
