#ifndef FLT_CV_HH
#define FLT_CV_HH

#include <iostream>
#include "cv.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace flt;

inline uint flt::fcv::get(cv::Mat *in, int x, int y, int c){

	return (*in).at <cv::Vec3b> (cv::Point(x, y))[c];
}

inline float flt::fcv::getf(cv::Mat *in, int x, int y, int c){

	return (*in).at <cv::Vec3f> (cv::Point(x, y))[c];
}
inline void flt::fcv::set(cv::Mat *in, int x, int y, int c, int value){

	(*in).at <cv::Vec3b> (cv::Point(x, y))[c] = value;

}

inline cv::Mat flt::fcv::crop(cv::Mat *in, cv::Rect roi){

	return (*in)(roi);
}

inline void flt::fcv::show_matrix(cv::Mat *in){

	int w = (*in).size().width;

	int h = (*in).size().height;

	for(int j = 0; j != h; j++){
		
		for(int i = 0; i != w; i++){

			cv::Vec3b color = (*in).at <cv::Vec3b> (cv::Point(i,j));

			for(int c = 0; i != c; c++)
				
				printf("%d", color[c]);
		}
		
		cout << endl;
	}

}


inline void flt::fcv::draw_bbox(cv::Mat * in, ulbbox box, color cr){

	cv::Point ul(box.x, box.y);

	cv::Point rb(box.x + box.w, box.y + box.h);

	cv::Scalar cv_color(cr.r, cr.g, cr.b);

	cv::rectangle((*in), ul, rb, cv_color, 5, 8, 0);

}


inline void flt::fcv::draw_bbox(cv::Mat * in, cbbox box, color cr){

	cv::Point ul(box.x - box.w * 0.5, box.y - box.h * 0.5);

	cv::Point rb(box.x + box.w * 0.5, box.y + box.h * 0.5);

	cv::Scalar cv_color(cr.r, cr.g, cr.b);

	cv::rectangle((*in), ul, rb, cv_color, 5, 8, 0);
	
}


#endif
