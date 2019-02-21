#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include "box.h"
#include "struct.h"


using namespace std;


class detector {

	public:

		detector(string & _json, string & _params, string & _mean,  string & _device, vector <string> & _classes, Size & _size, bool switching);
		
		~ detector();

		void convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes, Size & size_);

		int visualize(cv::Mat & in, vector <vector <bbox>> & bboxes);

		int detect(int tid);

		void detect_image(string & image);
		
		//int detect(cv::Mat & in);
		
		int capture(DetectType & dt, string & filename);

		int guard();

		void sw();

		int batch = 1;

		int cbatch = 1;

		int way = 1;

		bool switching = false;

		map <string, bool> alive;

		Size size, osize;
		
		Symbol net;

		float threshold = 0.5;

		vector <Executor *> E;

		string json, params;

		vector <string> classes;

		vector <cv::Mat> frames;

		vector <cv::Mat> input;
		
		vector <vector <cv::Mat>> tinput = vector <vector <cv::Mat>> (2);

		deque <cv::Mat> MatQueue;

		deque <cv::Mat> FrameQueue;
		
		deque <vector <vector <bbox>>> BoxesQueue;

		/*

		map <string, NDArray> args;

		map <string, NDArray> args2;

		map <string, NDArray> auxs;
		
		map <string, NDArray> auxs2;
		
		map <string, NDArray> grad;
		
		map <string, OpReqType> req;

		*/
		
		vector <map <string, NDArray>> args = vector <map <string, NDArray>> (2);

		vector <map <string, NDArray>> auxs = vector <map <string, NDArray>> (2);
		
		map <string, NDArray> nds;
		
		map <string, NDArray> grad;
		
		map <string, OpReqType> req;
		
		unsigned int wait = 100;
		
		Context * ctx;

		//map <string, vector <mutex>> lock;
		
		vector <mutex> lock = vector <mutex> (2);

		mutex mMatQueue, mBoxesQueue, mFrames;
		
		int post(bool v); // visualize

	private:

		int request();
		void unload(int no);
		void load(int no);

};

#endif
