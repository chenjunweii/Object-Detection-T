#ifndef FLT_DARKNET_IMAGE_H
#define FLT_DARKNET_IMAGE_H

#include <opencv2/opencv.hpp>


namespace flt{

	namespace darknet{
		
		struct image {
			
			int w, h, c;
			
			float * data;

		};

		void rgbgr_image(image im);

		image letterbox_image(image im, int w, int h);
		image load_image_color(char* filename,int w,int h);

	};
};

#endif
