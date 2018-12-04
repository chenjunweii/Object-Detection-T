#ifndef TVM_BOX_H
#define TVM_BOX_H

#include <vector>
#include "tvm.h"

using namespace std;

struct bbox {

	public:

		bbox();
		
		bbox(vector <float> & fbbox, Size & size);

		bbox(float c, float s, float x, float y, float x1, float y1);

		int c = 0;

		float s = 0;

		int x = 0;

		int y = 0;

		int x1 = 0;

		int y1 = 0;

};


#endif
