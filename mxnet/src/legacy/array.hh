#ifndef FLT_ARRAY_HH
#define FLT_ARRAY_HH


#include <iostream>
#include "shape.hh"

using namespace std;



template <typename T>

inline flt::farray::farray(T * t, flt::fshape s) : shape(s), data(t), rank(s.rank){};

template <typename T>

inline T flt::farray::operator [] (T i) {
	
	cursor = 0;

	rank_cursor = 0;

	if (rank == 1)
		
		return (*data)[i];
	

}


/* recursive */

template <typename T>

inline flt::farray flt::farray::operator [] (T i){

	if (i > shape[0])

		cout << "not allowed slice"
	
	cursor += shape[0] * i;

	rank_cursor += 1;

	return this;


}





#endif
