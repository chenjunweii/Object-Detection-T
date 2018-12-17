#ifndef DETECTOR_YOLOV3_DARKNET_H
#define DETECTOR_YOLOV3_DARKNET_H


#include <iostream>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include "executor.hh"
#include "box.hh"
#include <unistd.h>
#include <boost/format.hpp>
#include <condition_variable>
#include "darknet/yolo.h"


using namespace std;
using namespace cv;

namespace flt{

namespace detector{
	
	enum DetectType {
		video,
		image,
		camera
	};

	class yolov3 {

		public:

			inline yolov3(string network, string device, Size size, vector <string> & _classes,
				map <string, vector <int64_t>> & _det_in, map <string, vector <int64_t>> & _det_out);

			inline int capture_thread(DetectType & dt, string & filename);
			
			inline void detect_image(string sin);

			inline void detect();

			inline void pre();

			inline void post();

			inline void show();

			inline void preprocess(Mat & in);

			inline void letterbox(Mat & in);



			vector <string> classes;

			string network, device;
			
			map <string, vector <int64_t>> det_in, det_out;

			vector <string> det_out_node = {"reshape5_output", "reshape5_mask", "reshape5_bias", "reshape5_attr",
				"reshape3_output", "reshape3_mask", "reshape3_bias", "reshape3_attr",
				"reshape1_output", "reshape1_mask", "reshape1_bias", "reshape1_attr"};

			TVMExecutor * det;
			
			deque <Mat> OriginalQueue, ReSizedQueue;
			
			deque <vector <vector <flt::bbox>>> BoxesQueue;

			mutex mOriginalQueue, mResizedQueue, mBoxesQueue;

			DetectType detect_type;
			
			Size size, osize;
		
			map <string, bool> alive;

			vector <vector <float>> layerout;
			
			unsigned int wait = 100;

			inline ~ yolov3();

			inline void visualize(Mat & in, vector <vector <bbox>> & boxes, float threshold, int stay);
			
			inline void convert(float * fout, vector <vector <bbox>> & bboxes, Size & size_);

			inline int guard();

			inline void detect_thread();

			inline void post_thread(bool viz);

			inline void detect_video();

			inline void find_active(float *, float threshold);

			inline void get_box();

			inline void nms();
	
	};
	
}

}

#endif
