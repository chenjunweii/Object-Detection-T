#ifndef DETECTOR_SSD_H
#define DETECTOR_SSD_H

#include <iostream>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include "executor.hh"
#include "box.hh"
#include <unistd.h>
#include <boost/format.hpp>
#include <condition_variable>

using namespace std;
using namespace cv;

namespace flt {

namespace detector {

	enum DetectType {
		video,
		image,
		camera
	};
	
	class ssd {

		public:

			inline ssd(string network, string device, Size size, vector <string> & _classes,
				map <string, vector <int64_t>> & _det_in, map <string, vector <int64_t>> & _det_out, 
				map <string, vector <int64_t>> & _nms_out);

			inline int capture_thread(DetectType & dt, string & filename);
			
			inline void detect_image(string sin);

			inline void detect();

			inline void pre();

			inline void post();

			inline void show();

			vector <string> classes;

			string network, device;
			
			map <string, vector <int64_t>> det_in, det_out, nms_out;

			vector <string> det_out_node = {"cls_prob", "loc_preds", "anchor_boxes"};
			
			vector <string> nms_out_node = {"nms0_output"};
			
			TVMExecutor * det, * nms;
			
			deque <Mat> OriginalQueue, ReSizedQueue;
			
			deque <vector <vector <flt::bbox>>> BoxesQueue;

			mutex mOriginalQueue, mResizedQueue, mBoxesQueue;

			condition_variable cOriginalQueue, cResizedQueue, cBoxesQueue;

			DetectType detect_type;
			
			Size size, osize;
		
			map <string, bool> alive;
			
			unsigned int wait = 100;

			inline ~ssd();

			inline void visualize(Mat & in, vector <vector <bbox>> & boxes, float threshold, int stay);
			
			inline void convert(float * fout, vector <vector <bbox>> & bboxes, Size & size_);

			inline int guard();

			inline void detect_thread();

			inline void post_thread(bool viz);

			inline void detect_video();
	
	};

}

}


#endif
