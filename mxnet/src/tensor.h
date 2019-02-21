#ifndef FLT_TENSOR_H
#define FLT_TENSOR_H

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <random>
#include <stdexcept>

#include "src/shape.hh"

using namespace std;
using namespace mxnet::cpp;

namespace flt{

	class ftensor{
		
		private:
			
			float * data;
			
			int rank_cursor = 0;
			
			int cursor = 0;		
			
		public:

			fshape shape;
			
			int rank;
			
			inline ftensor(float * t, fshape s);

			template <typename T>

			inline float get1d(T t);

			template <typename T>
			
			inline int get(T t);

			template <typename T, typename ... Args>

			inline int get(T t, Args... arg);

			template <typename T, typename ... Args>

			inline float operator () (T t, Args... arg);
			
			template <typename T>
			
			inline float operator () (T t);

			inline void reshape(fshape s);

	};
}

#endif
