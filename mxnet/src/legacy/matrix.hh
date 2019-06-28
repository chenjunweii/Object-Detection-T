#ifndef FLT_MATRIX_HH
#define FLT_MATRIX_HH

#include <iostream>
#include "src/matrix.h"

using namespace std;

inline void flt::fmatrix::mgrid(int h, int w){

	float *x = new float[h, w];

	float *y = new float[h, w];

	for(int j = 0; j != h; j++)

	for(int i = 0; i != w; i++){

		x[h * i + w] =  i;
		
		y[h * i + w] =  j;

	}
}
#endif
