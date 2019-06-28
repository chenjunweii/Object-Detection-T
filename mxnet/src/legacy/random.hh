#ifndef FLT_RANDOM_HH
#define FLT_RANDOM_HH


#include <iostream>
#include <random>
#include <vector>
#include "random.h"

using namespace std;

inline void flt::frandom::uniform(float * array, float begin, float end, fshape shape){
	
	default_random_engine generator;
	
	uniform_real_distribution<float> uniform(begin, end);
	
	for(int i = 0; i != shape.size; ++i)

		array[i] = uniform(generator);



}

inline void flt::frandom::normal(float * array, float begin, float end, fshape shape){

	default_random_engine generator;

	normal_distribution<float> normal(begin, end)

	for(int i = 0; i != shape.size; ++i)

		array[i] = normal(generator);

}

#endif
