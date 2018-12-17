#ifndef DETECTOR_HH
#define DETECTOR_HH


#include "tvm.h"
#include "box.h"
#include "detector.h"
#include <mutex>
#include <unistd.h>
#include <boost/format.hpp>      

inline void convert(vector <float> & fout, vector <vector <flt::bbox>> & bboxes, Size & size_){

	vector <float> slice;
	for (int i = 0; i != 1; ++i){
			(bboxes).emplace_back(vector <flt::bbox> ());
		for (int j = 0; j != 5186; ++j){
			slice = vector <float> (fout.begin() + j * 6, fout.begin() + j * 6 + 6);
			if (slice[0] >= 0){
				(bboxes)[i].emplace_back(move(flt::bbox(slice, size_)));
			}
		}
	}
}

inline void convert(float * fout, vector <vector <flt::bbox>> & bboxes, Size & size_){

	vector <float> slice;
	for (int i = 0; i != 1; ++i){
			(bboxes).emplace_back(vector <flt::bbox> ());
		for (int j = 0; j != 5186; ++j){
			slice = vector <float> (fout + j * 6, fout + j * 6 + 6);
			if (slice[0] >= 0){
				(bboxes)[i].emplace_back(move(flt::bbox(slice, size_)));
			}
		}
	}
}
inline int visualize(Mat & in, vector <vector <flt::bbox>> & bboxes, vector <string> & classes, float threshold){
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

		cv::waitKey(int(0));

		//cv::waitKey(0);
}

#endif
