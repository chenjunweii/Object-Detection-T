#ifndef FLT_TENSOR_HH
#define FLT_TENSOR_HH

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <random>
#include <stdexcept>

#include "src/shape.hh"
#include "src/tensor.h"

using namespace std;
using namespace mxnet::cpp;

inline flt::ftensor::ftensor(float * t, flt::fshape s) : shape(s), data(t), rank(s.rank){};

template <typename T>

inline float flt::ftensor::get1d(T t){
	
	if (rank > 1)

		throw invalid_argument("not allowed slice");

	return data[t];

}

template <typename T>

inline int flt::ftensor::get(T t){
	
	//cursor = 0;

	return t;

}
template <typename T, typename ... Args>

inline int flt::ftensor::get(T t, Args... arg){
	
	if (t >= shape[rank - sizeof...(arg) - 1])
		
		throw std::invalid_argument( "Slice is not allowed" );

	return get(arg...) + shape[rank - (sizeof...(arg))] * t;
	
}

template <typename T>

inline float flt::ftensor::operator () (T t){

	get1d(t);
}

template <typename T, typename ... Args>

inline float flt::ftensor::operator () (T t, Args ... arg) {
	
	return data[get(t, arg...)];

}

inline void flt::ftensor::reshape(fshape s){

	shape = s;
}

#endif

