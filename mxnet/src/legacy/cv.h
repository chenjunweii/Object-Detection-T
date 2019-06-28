#ifndef FLT_CV_H
#define FLT_CV_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

namespace flt{

	struct cbbox{

		float x;

		float y;

		float w;

		float h;

	};

	struct ulbbox{
		
		float x;

		float y;

		float w;

		float h;
		
	};

	struct color{

		int r;

		int g;

		int b;

	};

	namespace fcv{

		inline void show_matrix(cv::Mat *);

		inline uint get(cv::Mat *, int x, int y, int c);
		
		inline float getf(cv::Mat *, int x, int y, int c);
		
		inline void set(cv::Mat *, int x, int y, int c, int value);

		inline cv::Mat crop(cv::Mat *in, cv::Rect roi);

		inline void draw_bbox(cv::Mat *, ulbbox, color cr = {255, 255, 0});
		
		inline void draw_bbox(cv::Mat *, cbbox, color cr = {255, 255, 0});


	}
}


#endif
