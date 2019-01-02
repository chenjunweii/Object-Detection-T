#ifndef FLT_TRT_BOX_HH
#define FLT_TRT_BOX_HH

#include "utils.hh"


namespace flt {

	inline int clip(int x, int lower, int upper) {
  
		return max(lower, min(x, upper));
	
	}

struct bbox {

	public:

		int c = 0;

		float s = 0;

		int x = 0;

		int y = 0;

		int x1 = 0;

		int y1 = 0;
		
		bbox(){};
		
		bbox(float * fout, Size & size){

			c = int(fout[1]); s = fout[2];

			int w = size.width;

			int h = size.height;
			
			x = clip(int(fout[3] * w), 0, w); y = clip(int(fout[4] * h), 0, h);
			
			x1 = clip(int(fout[5] * w), 0, w); y1 = clip(int(fout[6] * h), 0, h);
		}

		bbox(vector <float> & fout, Size & size){

			c = int(fout[1]); s = fout[2];

			int w = size.width;

			int h = size.height;
			
			x = clip(int(fout[3] * w), 0, w); y = clip(int(fout[4] * h), 0, h);
			
			x1 = clip(int(fout[5] * w), 0, w); y1 = clip(int(fout[6] * h), 0, h);
		}

		bbox(float c, float s, float x, float y, float x1, float y1){

		}

};

typedef vector <bbox> bboxes;

};

#endif

