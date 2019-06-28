#ifndef DETECTOR_YOLO3_HH
#define DETECTOR_YOLO3_HH

#include <memory>
#include <assert.h>
#include "struct.h"
#include "box.hh"
#include <omp.h>
#include "utils.hh"
#include <math.h>
#include <mutex>
#include <unistd.h>
#include <boost/format.hpp>

#include "src/detector/yolo3.h"

#include "src/darknet/image.hh"

using namespace std;

using namespace flt::mx::image;

using namespace flt::darknet;

yolo3::yolo3(string & _json, string & _params, string & _device, vector <string> & _classes, int short_edge, cv::Size & frame_size, bool _streaming, bool _switching) : classes(_classes), switching(_switching), streaming(_streaming) {

	if (_device.compare("gpu") == 0)
		ctx = new Context(DeviceType::kGPU, 0);
	else
		ctx = new Context(DeviceType::kCPU, 0);
	
	net = Symbol::Load(_json);

	size = find_detection_size(short_edge, frame_size); // == dsize

	dheight = size.height; dwidth = size.width;

	dfsize = dheight * dwidth * dchannel;

	map <string, NDArray> ndc = NDArray::LoadToMap(_params); // ndarray cpu

	vector <float> _mean (dfsize); vector <float> _std (dfsize);

	int v = dheight * dwidth;

	for (int _c = 0; _c != 3; ++_c){

		for (int _v = 0; _v != v; ++_v){

			_mean[_c * v + _v] = mean[_c];

			_std[_c * v + _v] = std[_c];

		}
	}

	nds["mean"] = NDArray(Shape(1, 3, dheight, dwidth), *ctx);

	nds["std"] = NDArray(Shape(1, 3, dheight, dwidth), *ctx);

	nds["mean"].SyncCopyFromCPU(_mean.data(), dfsize);
	
	nds["std"].SyncCopyFromCPU(_std.data(), dfsize);

	NDArray::WaitAll();

	cout << "size height : " << size.height << endl;

	cout << "size width : " << size.width << endl;

	if (streaming)

	//streamer = cv::VideoWriter("appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=superfast ! rtph264pay ! tcpserversink host=0.0.0.0 port=5000 recover-policy=keyframe sync-method=latest-keyframe sync=false", cv::CAP_GSTREAMER, 0, 15, cv::Size(1280, 720), true);
	//
	//
	streamer = cv::VideoWriter("appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=superfast ! mpegtsmux ! queue ! tcpserversink host=0.0.0.0 port=5000 recover-policy=keyframe sync-method=latest-keyframe sync=true", cv::CAP_GSTREAMER, 0, 30, cv::Size(1280, 720), true);

	
	fdata.emplace_back(move(vector <float> (dfsize)));
	args[0]["data"] = NDArray(Shape(batch, 3, size.height, size.width), *ctx);
	if (switching){
		args[1]["data"] = NDArray(Shape(batch, 3, size.height, size.width), *ctx);
		fdata.emplace_back(move(vector <float> (dfsize)));

	}

	for (auto & k : ndc){
		auto type = k.first.substr(0, 3);
		auto node = k.first.substr(4);
		if (type.compare("arg") == 0){
			args[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&args[0][node]);
			//NDArray::WaitAll();
			//req[node] = OpReqType::kNullOp;
			if (switching)
				args[1][node] = args[0][node];
		}
		else if (type.compare("aux") == 0){
			auxs[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&auxs[0][node]);
			//NDArray::WaitAll();
			if (switching)
				auxs[1][node] = auxs[0][node];
		}
	}
	NDArray::WaitAll();

	if (switching)
		way = 2;

	for (int i = 0; i != way; i++)
		E.emplace_back(net.SimpleBind(*ctx, args[i], grad, req, auxs[i]));

}


inline int yolo3::guard(){
	for (auto & i : alive){
		if (i.second == false){
			cout << "[*] Guard Close " << i.first << endl;
			for (auto & j : alive){
				j.second = false;
			}
		}
	}
}


inline int yolo3::detect_video(string & filename){

	cv::Mat resized;
	cv::Mat frame;
	cv::VideoCapture capture;
	capture = cv::VideoCapture(filename);
	//else if (dt == DetectType::camer)
	if (!capture.isOpened())
		return -1;
//	capture.set(CV_CAP_PROP_FRAME_WIDTH,720);
//	capture.set(CV_CAP_PROP_FRAME_HEIGHT,360);
	int counter_sys = 0;

	owidth = int(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	
	oheight = int(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	
	osize = cv::Size(owidth, oheight);

//vector <float> fdata(dfsize);
	
	auto t_start_sys = std::chrono::high_resolution_clock::now();
		
	vector <vector <bbox>> boxes;
    
	while(true){
	
		capture >> frame;
		
		if(frame.empty())
		
			break;
		
		cv::resize(frame, resized, cv::Size(size.width, size.height), CV_8UC3, 0, 2);
		
		cv::cvtColor(resized, resized, CV_BGR2RGB);
		
		auto t_start_inf = std::chrono::high_resolution_clock::now();
		
		load(resized, args[0]["data"], fdata[0]);
		
		E[0]->Forward(false);

		convert(E[0]->outputs, boxes, size, osize);
		
		auto t_end_inf = std::chrono::high_resolution_clock::now();
		
		float total_inf = std::chrono::duration <float, std::milli> (t_end_inf - t_start_inf).count();
		
		cout << "Inference Time Elapse Inference: " << total_inf << endl;
		
		visualize(frame, boxes, false);

		counter_sys += 1;
		
		if (counter_sys >= 100){

			auto t_end_sys = std::chrono::high_resolution_clock::now();
			
			float total_sys = std::chrono::duration <float, std::milli> (t_end_sys - t_start_sys).count();
			
			cout << "System Time Elapse Inference: " << total_sys << endl;

			break;

		}

		if (streaming)

			streamer << frame;

		boxes.clear();

		boxes.shrink_to_fit();
	}

}
inline int yolo3::capture(DetectType & dt, string & filename){
	cv::Mat resized;
	cv::Mat frame;
	cv::VideoCapture capture;
	if (dt == DetectType::video)
		capture = cv::VideoCapture(filename);
	//else if (dt == DetectType::camer)
	if (!capture.isOpened())
		return -1;

	osize = cv::Size(int(capture.get(CV_CAP_PROP_FRAME_WIDTH)), int(capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

	bool get = false;
	
	alive["capture"] = true;

	int frame_count = -1;

	cv::Mat fake(1, 1, CV_8UC3);

    while(true){

		cout << "in caputre " << endl;

		while(FrameQueue.size() > 5)
			usleep(wait * 1);

		capture >> frame;

		frame_count += 1;

		if(frame.empty())

			break;

		if (frame_count % frame_sample == 0){

			cv::resize(frame, resized, cv::Size(size.width, size.height), CV_8UC3, 0, 2);
			
			cv::cvtColor(resized, resized, CV_BGR2RGB);

			emplace <cv::Mat> ((resized), MatQueue, mMatQueue, get, wait, false);
		}

		else{

			emplace <cv::Mat> (fake, MatQueue, mMatQueue, get, wait, true);

		}

		emplace <cv::Mat> ((frame), FrameQueue, mFrames, get, wait, false);
		
		if (not alive["capture"])
			break;

		usleep(wait);
    }
	alive["capture"] = false;
    return 0;	
}

inline int yolo3::post(bool viz){

	bool get = false;
	alive["capture"] = true;

	int frame_count = 0;

	cv::Mat frame;

	vector <vector <bbox>> boxes;
	while (true){
		boxes.clear();
		boxes.shrink_to_fit();

		while (! get){
			while (BoxesQueue.empty())
				usleep(wait);
			get = mBoxesQueue.try_lock();
		}

		boxes = move(BoxesQueue.front());
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

		visualize(frame, boxes, false);
		if (streaming)
			streamer << frame;
		guard();
		usleep(wait);
		if (not alive["capture"])
			break;
	}
}

inline void yolo3::load(int no){
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
	load(tinput[no][0], args[no]["data"], fdata[no]);

	tinput[no].clear();
	tinput[no].shrink_to_fit();
	//usleep(wait);
}


inline void yolo3::unload(int no){
	lock[no].unlock();
}

inline int yolo3::request(){
	while (true){
		for (int j = 0; j != way; ++j)
			if (lock[j].try_lock())
				return j;
		usleep(wait);
	}
}

inline int yolo3::detect(int tid){
	int no;
	bool get = false;
	string stid = to_string(tid);
	vector <vector <bbox>> boxes;
	int frame_count = -1;
	alive[stid] = true;

	cv::Mat frame(size.width, size.height, CV_8UC3);
	
	auto t_start_sys = std::chrono::high_resolution_clock::now();

	int sys_counter = 0;

	while (true){

		acquire(frame, MatQueue, mMatQueue, get, wait, "detection");

		cv::Size frame_size = frame.size();

		if (frame_size.width == 1 and frame_size.height == 1){

			cout << "m" << endl;
		}

		else {
			
			boxes.clear();
			
			boxes.shrink_to_fit();
			
			no = request();
			
			auto t_start_inf = std::chrono::high_resolution_clock::now();

			load(frame, args[no]["data"], fdata[0]);

			cout << "Detect Thread " << no << endl;
			
			E[no]->Forward(false);

			convert(E[no]->outputs, boxes, size, osize);

			auto t_end_inf = std::chrono::high_resolution_clock::now();
			
			float total_inf = std::chrono::duration <float, std::milli> (t_end_inf - t_start_inf).count();
			
			cout << "Inference Time Elapse Inference: " << total_inf << endl;

			unload(no);
		}

		sys_counter += 1;

		if (sys_counter >= 100){

			auto t_end_sys = std::chrono::high_resolution_clock::now();
			
			float total_sys = std::chrono::duration <float, std::milli> (t_end_sys - t_start_sys).count();
			
			cout << "AVG Time Elapse Inference: " << total_sys << endl;

		}
		while (! get){
			get = mBoxesQueue.try_lock();
			if (! get)
				usleep(wait);
		}

		BoxesQueue.push_back((boxes));
		mBoxesQueue.unlock();
		usleep(wait);
		if (not alive[stid])
			break;
	}

	alive[stid] = false;
}

yolo3::~yolo3(){
	delete ctx;
	for (auto & e : E)
		delete e;
}

//inline void yolo3::convert(vector <NDArray> & ndout, vector <vector <float>> & classes, vector <vector <float>> & scores, vector <vector <bbox>> & bboxes, cv::Size & size_){
inline void yolo3::convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes, cv::Size & dsize, cv::Size & osize){

	if (classes_size == 0){

		classes_shape = ndout[0].GetShape();
		scores_shape = ndout[1].GetShape();
		bboxes_shape = ndout[2].GetShape();
		
		classes_size = classes_shape[0] * classes_shape[1] * classes_shape[2];
		scores_size = scores_shape[0] * scores_shape[1] * scores_shape[2];
		bboxes_size = bboxes_shape[0] * bboxes_shape[1] * bboxes_shape[2];
	
		fclasses = vector <float> (classes_size);
		fscores = vector <float> (scores_size);
		fbboxes = vector <float> (bboxes_size);

	}
	
	ndout[0].SyncCopyToCPU(fclasses.data(), classes_size);
	ndout[1].SyncCopyToCPU(fscores.data(), scores_size);
	ndout[2].SyncCopyToCPU(fbboxes.data(), bboxes_size);

	vector <float> slice;
	for (int i = 0; i != bboxes_shape[0]; ++i){
		(bboxes).emplace_back(vector <bbox> ());
		for (int j = 0; j != bboxes_shape[1]; ++j){
			if (fclasses[j] >= 0 and fscores[j] > threshold){

				slice = vector <float> (fbboxes.begin() + j * 4, fbboxes.begin() + j * 4 + 4);
				
				(bboxes)[i].emplace_back(move(bbox(fclasses[j], fscores[j], slice, dsize, osize)));
			}
		}
	}
	
}

inline int yolo3::visualize(cv::Mat & in, vector <vector <bbox>> & bboxes, bool show){

		assert(bboxes.size() == 1); // assert batch size = 1
		for (auto & box : bboxes[0]){
			//if (box.s >= threshold){
				cv::Point ul(box.x, box.y);
				cv::Point br(box.x1, box.y1);
				cv::Point tp(box.x1, box.y1 - 5);
				cv::rectangle(in, ul, br, cv::Scalar(0, 255, 0), 3.5);
				string text = boost::str(boost::format("%s : %f") % classes[box.c] % box.s);
				cv::putText(in, text, tp, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,  cv::Scalar(0, 0, 255, 255), 2);
				cout << "classes : " << classes[box.c] << endl;
			//}
		}

		//cout << in.size().width << endl;
		//cout << in.size().height << endl;
		//
		if (show){
			cv::imshow("Visualize", in);
			cv::waitKey(int(1000.0 / 30));
		}
		//cv::waitKey(0);
}


inline cv::Size yolo3::find_detection_size(int short_edge, cv::Size & _size, int max_size, int mult_base){

	int h = _size.height; int w = _size.width;

	int im_size_min, im_size_max;

	if (w > h){

		im_size_min = h; im_size_max = w;

	}

	else{

		im_size_min = w; im_size_max = h;
	}
    
	float scale = float(short_edge) / float(im_size_min);
    
	if (round(scale * im_size_max / mult_base) * mult_base > max_size)
        scale = float(floor(max_size / mult_base) * mult_base) / float(im_size_max);

    int new_w = int(round(w * scale / mult_base) * mult_base);
    
	int new_h = int(round(h * scale / mult_base) * mult_base);

    return cv::Size(new_w, new_h);

}

inline void yolo3::load(cv::Mat & mat, NDArray & nd, vector <float> & farray){

	int cbase, hbase, chbase = 0;

	#pragma omp parallel for

	for (int _c = 0; _c < dchannel; ++_c) {

		cbase = _c * dheight * dwidth;

		float _mean = mean[_c]; float _std = std[_c];

		for (int _h = 0; _h < dheight; ++_h) {

			hbase = _h * dwidth;

			chbase = cbase + hbase;

			for (int _w = 0; _w < dwidth; ++_w) {
		  		
				//farray.emplace_back(static_cast <float> (mat.data[(i * h + j) * channel + c]));
				//farray[cbase + hbase + _w] = (((static_cast <int> (mat.data[(hbase + _w) * dchannel + _c])) / 255.0) - _mean) / _std;
				farray[chbase + _w] = static_cast <int> (mat.data[(hbase + _w) * dchannel + _c]);
			}
		}
	}

	nd.SyncCopyFromCPU(farray.data(), dfsize);

	nd /= 255.0;
	
	nd -= nds["mean"];

	nd /= nds["std"];

	nd.WaitToRead();

}



inline int yolo3::detect_image(cv::Mat & in){

	int t = 0; // thread id
	
	osize = in.size();

	cv::Mat resized, rgb;

	cv::cvtColor(in, rgb, CV_BGR2RGB);
	
	cv::resize(rgb, resized, cv::Size(size.width, size.height), CV_8UC3, 0, 2);

	load(resized, args[t]["data"], fdata[0]);

	tinput[t].clear();
	
	tinput[t].shrink_to_fit();
		
	E[t]->Forward(false);
	
	vector <vector <bbox>> boxes;
	
	convert(E[t]->outputs, boxes, size, osize);
	
	visualize(in, boxes, true);
}

#endif
