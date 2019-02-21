#ifndef FLT_STL_H
#define FLT_STL_H

#include <iostream>
#include <vector>


using namespace std;

namespace flt{

	namespace fvector{

		inline vector <int> slice(vector <int>, int, int);
		inline vector <int> concat(vector <int>, vector <int>);
		inline vector <float> arange(float vmin, float vmax, float step);
		inline void print(vector <string> &v, bool p);

	}

}

#endif
