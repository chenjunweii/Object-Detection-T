#ifndef FLT_DARKNET_IMAGE_H
#define FLT_DARKNET_IMAGE_H

#include "yolo.h"
#include <opencv2/opencv.hpp>

//using namespace cv;

namespace flt{

	namespace darknet{

	struct image {
		
		int w, h, c;
		
		float * data;

	};

	image letterbox_cv2(image im, int w, int h);
	
	image load_image_color(char * filename, int w, int h);

	void free_image(image m);

	void rgbgr_image(image im);

	void ipl_into_image(IplImage * src, image im);

	image letterbox_image(image im, int w, int h);

	static float get_pixel(image m, int x, int y, int c);

	static void set_pixel(image m, int x, int y, int c, float val);

	static void add_pixel(image m, int x, int y, int c, float val);

	}
}

#endif
