#ifndef FLT_RANDOM_H
#define FLT_RANDOM_H

#include <iostream>
#include <random>
#include <vector>
#include "shape.h"

using namespace std;

namespace flt{

	namespace frandom{
		
		inline void uniform (float * array, float begin, float end, fshape shape);

		inline void normal (float * array, float begin, float end, fshape shape); 
	}
}



#endif
