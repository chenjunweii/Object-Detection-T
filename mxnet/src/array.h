#ifndef FLT_ARRAY_H
#define FLT_ARRAY_H

#include <iostream>
#include "shape.h"

using namespace std;

namespace flt{

	class farray{
		
		constexpr fshape shape;
		constexpr int rank;
		private:

			template <typename T>
			
			T * data;
			
			int rank_cursor = 0;
			int cursor = 0;		
		public:

			template <typename T>

			farray(T * );

	}
}

#endif
