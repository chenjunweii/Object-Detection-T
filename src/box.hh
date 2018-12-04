#include "box.h"
#include "utils.hh"

bbox::bbox(){};

bbox::bbox(vector <float> & fout, Size & size){

	c = int(fout[0]); s = fout[1];

	int w = size.width;

	int h = size.height;
	
	x = clip(int(fout[2] * w), 0, w); y = clip(int(fout[3] * h), 0, h);
	
	x1 = clip(int(fout[4] * w), 0, w); y1 = clip(int(fout[5] * h), 0, h);
}

bbox::bbox(float c, float s, float x, float y, float x1, float y1){

}

