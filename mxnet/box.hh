#include "box.h"
#include "utils.hh"

bbox::bbox(){};

bbox::bbox(vector <float> & fout, cv::Size & size){

	// ssd box
	//
	c = int(fout[0]); s = fout[1];
	
	x = clip(int(fout[2] * size.width), 0, size.width); y = clip(int(fout[3] * size.height), 0, size.height);
	
	x1 = clip(int(fout[4] * size.width), 0, size.width); y1 = clip(int(fout[5] * size.height), 0, size.height);
}

bbox::bbox(float c, float s, float x, float y, float x1, float y1){

}


bbox::bbox(float klass, float score, vector <float> & _bbox, cv::Size & dsize, cv::Size & osize){

	// yolo box
	//
	// dsize : detection size 
	//
	// osize : original size (captured frame size)

	c = klass; s = score;

	float rw = float(osize.width) / dsize.width;
	
	float rh = float(osize.height) / dsize.height;

	//x = clip(int(_bbox[0] * size.width), 0, size.width); y = clip(int(_bbox[1] * size.height), 0, size.height);
	
	//x1 = clip(int(_bbox[2] * size.width), 0, size.width); y1 = clip(int(_bbox[3] * size.height), 0, size.height);
	//
	//
	x = int(_bbox[0] * rw); y = int(_bbox[1] * rh); x1 = int(_bbox[2] * rw); y1 = int(_bbox[3] * rh);
}
