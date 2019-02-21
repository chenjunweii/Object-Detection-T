#ifndef FLT_SHAPE_HH
#define FLT_SHAPE_HH

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <random>
#include "shape.h"

using namespace std;
using namespace mxnet::cpp;

flt::fshape & flt::fshape::operator = (fshape & instance){
	
	this->v = instance.v;

	this->rank = instance.rank;

	return (*this);
}

int flt::fshape::operator [] (int i) {
			
	return v[i];
		
}


/*ostream & operator << (ostream & stream, flt::fshape const & instance){
	
	cout << "[";
	
	for(int i = 0; i != instance.rank; ++i){

		stream << instance.v[i];

		if(i != instance.rank - 1)

			cout << " ";
	}

	cout << "]";

	stream << endl;

	return stream;
}*/

template <typename T>

inline flt::fshape::fshape(T t) : rank(1) {			
	
	size = 1;

	construct(t);

}


template <typename T, typename... Args>

inline flt::fshape::fshape(T t, Args... args) : rank(sizeof...(args) + 1) {			
	
	size = 1;

	construct(t, args...);
}

template <typename T>

inline void flt::fshape::construct(T t){
	
	v.push_back(t);
	
	size *= t;
}

template <typename T, typename... Args>

inline void flt::fshape::construct(T t, Args... args){
	
	v.push_back(t);
	
	size *= t;

	construct(args...);

}

#endif
