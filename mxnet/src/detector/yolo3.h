#ifndef DETECTOR_YOLO3_H
#define DETECTOR_YOLO3_H

#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include "box.h"
#include "struct.h"

using namespace std;

class yolo3 {

	public:

		yolo3(string & _json, string & _params, string & _device, vector <string> & _classes, int short_edge, cv::Size & frame_size, bool streaming, bool switching);
		
		~ yolo3();

		void convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes, cv::Size & dsize, cv::Size & osize);

		int visualize(cv::Mat & in, vector <vector <bbox>> & bboxes, bool show);

		int detect(int tid);

		int detect_video(string & filename);

		void load(cv::Mat & mat, NDArray & nd, vector <float> & farray);

		vector <float> mean = {0.485, 0.456, 0.406};

		vector <float> std = {0.229, 0.224, 0.225};

		void detect_image(string & image);
		
		//int detect(cv::Mat & in);
		//
		//
		int detect_image(cv::Mat & in);
		
		int capture(DetectType & dt, string & filename);

		cv::Size find_detection_size(int short_edge, cv::Size & _size, int max_size = 1024, int mult_base = 1);

		int guard();

		vector <vector <float>> fdata;

		void sw();

		int batch = 1;

		int dwidth, dheight, owidth, oheight;

		int dchannel = 3; int ochannel = 3;

		int dfsize, ofsize; // float data size

		int cbatch = 1;

		int way = 1;

		bool switching = false;

		bool streaming = false;

		map <string, bool> alive;

		cv::Size size, osize; // size => detection input size, osize => captured original frame size
		
		Symbol net;

		float threshold = 0.5;

		int frame_sample = 1;

		cv::VideoWriter streamer;

		vector <Executor *> E;

		string json, params;

		vector <string> classes;

		vector <cv::Mat> frames;

		vector <cv::Mat> input;
		
		vector <vector <cv::Mat>> tinput = vector <vector <cv::Mat>> (2);

		deque <cv::Mat> MatQueue;

		deque <cv::Mat> FrameQueue;
		
		deque <vector <vector <bbox>>> BoxesQueue;
		
		vector <map <string, NDArray>> args = vector <map <string, NDArray>> (2);

		vector <map <string, NDArray>> auxs = vector <map <string, NDArray>> (2);
		
		map <string, NDArray> nds;
		
		map <string, NDArray> grad;
		
		map <string, OpReqType> req;
		
		unsigned int wait = 5;
		
		Context * ctx;

		//map <string, vector <mutex>> lock;
		
		vector <mutex> lock = vector <mutex> (2);

		mutex mMatQueue, mBoxesQueue, mFrames;
		
		int post(bool v); // visualize
		vector <mx_uint> classes_shape;// = nullptr;
		vector <mx_uint> scores_shape;// = nullptr;
		vector <mx_uint> bboxes_shape;// = nullptr;
		int classes_size = 0;//classes_shape[0] * classes_shape[1] * classes_shape[2];
		int scores_size = 0;//scores_shape[0] * scores_shape[1] * scores_shape[2];
		int bboxes_size = 0;//bboxes_shape[0] * bboxes_shape[1] * bboxes_shape[2];
		vector <float> fclasses;
		vector <float> fscores;
		vector <float> fbboxes;

	private:

		int request();
		void unload(int no);
		void load(int no);

};


#endif
