#include <memory>
#include <assert.h>
#include "struct.h"
#include "box.hh"
#include "utils.hh"
#include "detector.h"
#include <mutex>
#include <unistd.h>
#include <boost/format.hpp>

using namespace std;
using namespace flt::mx::image;

detector::detector(string & _json, string & _params, string & _mean, string & _device, vector <string> & _classes, Size & _size, bool _switching) : classes(_classes), size(_size), switching(_switching) {

	if (_device.compare("gpu") == 0)
		ctx = new Context(DeviceType::kGPU, 0);
	else
		ctx = new Context(DeviceType::kCPU, 0);
	
	if (switching)
		way = 2;
	else
		way = 1;
	
	net = Symbol::Load(_json);
	
	map <string, NDArray> ndc = NDArray::LoadToMap(_params); // ndarray cpu
	map <string, NDArray> ndm = NDArray::LoadToMap(_mean); // ndarray mean

	nds["mean"] = NDArray(Shape(1, 3, 1, 1), *ctx);
	ndm["mean"].CopyTo(&nds["mean"]);
	
	NDArray::WaitAll();

	args[0]["data"] = NDArray(Shape(batch, 3, _size.h, _size.w), *ctx);
	if (switching)
		args[1]["data"] = NDArray(Shape(batch, 3, _size.h, _size.w), *ctx);

	for (auto & k : ndc){
		auto type = k.first.substr(0, 3);
		auto node = k.first.substr(4);
		if (type.compare("arg") == 0){
			args[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&args[0][node]);
			NDArray::WaitAll();
			//req[node] = OpReqType::kNullOp;
			if (switching)
				args[1][node] = args[0][node];
		}
		else if (type.compare("aux") == 0){
			auxs[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&auxs[0][node]);
			NDArray::WaitAll();
			if (switching)
				auxs[1][node] = auxs[0][node];
		}
	}
	NDArray::WaitAll();

	for (int i = 0; i != way; i++)
		E.emplace_back(net.SimpleBind(*ctx, args[i], grad, req, auxs[i]));
}

inline int detector::guard(){
	for (auto & i : alive){
		if (i.second == false){
			cout << "[*] Guard Close " << i.first << endl;
			for (auto & j : alive){
				j.second = false;
			}
		}
	}
}

inline int detector::capture(DetectType & dt, string & filename){
	cv::Mat resized;
	cv::Mat frame;
	cv::VideoCapture capture;
	if (dt == DetectType::video)
		capture = cv::VideoCapture(filename);
	//else if (dt == DetectType::camer)
	if (!capture.isOpened())
		return -1;

	osize = Size(int(capture.get(CV_CAP_PROP_FRAME_WIDTH)), int(capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

	bool get = false;
	alive["capture"] = true;
    while(true){
		while(FrameQueue.size() > 15)
			usleep(wait * 3);
		capture >> frame;
		if(frame.empty())
			break;
		cv::resize(frame, resized, cv::Size(size.w, size.h));
		//frames.emplace_back(frame);
		cv::cvtColor(resized, resized, CV_BGR2RGB);
		get = false;
		while (! get){
			get = mFrames.try_lock();
			if (not get)
				usleep(wait);
		}
		FrameQueue.push_back(move(frame));
		mFrames.unlock();

		get = false;
		while (! get){
			get = mMatQueue.try_lock();
			if (not get)
				usleep(wait);
		}
		MatQueue.push_back(move(resized));
		mMatQueue.unlock();
		usleep(wait);
		if (not alive["capture"])
			break;
    }
	alive["capture"] = false;
    return 0;	
}

inline int detector::post(bool viz){

	bool get = false;
	alive["capture"] = true;
	cv::Mat frame;
	while (true){
		while (! get){
			while (BoxesQueue.empty())
				usleep(wait);
			get = mBoxesQueue.try_lock();
		}

		vector <vector <bbox>> boxes = move(BoxesQueue.front());
		BoxesQueue.pop_front();
		BoxesQueue.shrink_to_fit();
		mBoxesQueue.unlock();
		get = false;

		while (! get){
			get = mFrames.try_lock();
			usleep(wait);
		}

		get = false;
		frame = move(FrameQueue.front());
		FrameQueue.pop_front();
		FrameQueue.shrink_to_fit();
		mFrames.unlock();
		if (viz)
			visualize(frame, boxes);
		guard();
		usleep(wait);
		if (not alive["capture"])
			break;
	}
}

inline void detector::load(int no){
	bool get = false;
	while (! get){
		while (MatQueue.empty()){
			usleep(wait);
		}
		get = mMatQueue.try_lock();
	}
	cv::Mat frame = move(MatQueue.front());
	MatQueue.pop_front();
	MatQueue.shrink_to_fit();
	mMatQueue.unlock();
	tinput[no].emplace_back(move(frame));
	MatVector_to_NDArray(args[no]["data"], tinput[no], *ctx);
	args[no]["data"] -= nds["mean"];//args[no]["data"];
	NDArray::WaitAll();

	tinput[no].clear();
	tinput[no].shrink_to_fit();
	usleep(wait);
}


inline void detector::unload(int no){
	lock[no].unlock();
}

inline int detector::request(){
	while (true){
		for (int j = 0; j != way; ++j)
			if (lock[j].try_lock())
				return j;
		usleep(wait);
	}
}

inline int detector::detect(int tid){
	int no;
	bool get = false;
	string stid = to_string(tid);
	vector <vector <bbox>> boxes;
	alive[stid] = true;
	while (true){
		no = request();
		load(no);
		get = false;
		E[no]->Forward(false);

		convert(E[no]->outputs, boxes, osize);
		while (! get){
			get = mBoxesQueue.try_lock();
			if (! get)
				usleep(wait);
		}

		BoxesQueue.push_back(move(boxes));
		mBoxesQueue.unlock();
		unload(no);
		boxes.clear();
		boxes.shrink_to_fit();
		usleep(wait);
		if (not alive[stid])
			break;
	}

	alive[stid] = false;
}

detector::~detector(){
	delete ctx;
	for (auto & e : E)
		delete e;
}

inline void detector::convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes, Size & size_){

	vector <mx_uint> shape = ndout[0].GetShape(); // batach, ndets, 6
	int fsize = shape[0] * shape[1] * shape[2];
	vector <float> fout (fsize);
	ndout[0].SyncCopyToCPU(fout.data(), fsize);
	NDArray::WaitAll();
	vector <float> slice;
	for (int i = 0; i != shape[0]; ++i){
		(bboxes).emplace_back(vector <bbox> ());
		for (int j = 0; j != shape[1]; ++j){
			slice = vector <float> (fout.begin() + j * 6, fout.begin() + j * 6 + 6);
			if (slice[0] >= 0){
				(bboxes)[i].emplace_back(move(bbox(slice, size_)));
			}
		}
	}
}

inline int detector::visualize(cv::Mat & in, vector <vector <bbox>> & bboxes){

		assert(bboxes.size() == 1); // assert batch size = 1
		for (auto & box : bboxes[0]){
			if (box.s >= threshold){
				cv::Point ul(box.x, box.y);
				cv::Point br(box.x1, box.y1);
				cv::Point tp(box.x1, box.y1 - 5);
				cv::rectangle(in, ul, br, cv::Scalar(0, 255, 0), 3.5);
				string text = boost::str(boost::format("%s : %f") % classes[box.c] % box.s);
				cv::putText(in, text, tp, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,  cv::Scalar(0, 0, 255, 255), 2);
				cout << "classes : " << classes[box.c] << endl;
			}
		}
		cv::imshow("Visualize", in);
		cv::waitKey(int(1000.0 / 30));
		//cv::waitKey(0);
}


