#include "box.h"
#include "utils.hh"

bbox::bbox(){};

bbox::bbox(vector <float> & fout, Size & size){

	c = int(fout[0]); s = fout[1];
	
	x = clip(int(fout[2] * size.w), 0, size.w); y = clip(int(fout[3] * size.h), 0, size.h);
	
	x1 = clip(int(fout[4] * size.w), 0, size.w); y1 = clip(int(fout[5] * size.h), 0, size.h);
}

bbox::bbox(float c, float s, float x, float y, float x1, float y1){

}

