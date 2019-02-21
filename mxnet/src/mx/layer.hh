#ifndef LAYER_CC
#define LAYER_CC

#include <iostream>
#include <vector>
#include <string>
#include <mxnet-cpp/MxNetCpp.h>
#include "layer.h"

using namespace std;
using namespace mxnet::cpp;

inline void flt::mx::layer::conv(char * prefix,
	char * name, 
	char * input,
	map <string, Symbol> *neurons,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	int filter,
	Shape kernel,
	Shape stride,
	Shape pad,
	bool isRelu,
	bool noBias){

	string sw = string("w") + name;
	
	string sb = string("b") + name;

	string sr = string("r") + name;

	string in = string(prefix) + input;

	string out = string(prefix) + name;
	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);
	}

	if(isRelu)

		(*neurons)[out] = relu(sr, Convolution(out, (*neurons)[in], (*weight)[sw], (*bias)[sb], kernel, filter, stride, Shape(1,1), pad));

	else

		(*neurons)[out] = relu(Convolution(out, (*neurons)[in], (*weight)[sw], (*bias)[sb], kernel, filter, stride, Shape(1,1), pad));

}

inline void flt::mx::layer::conv(char * name, 
	char * input,
	map <string, Symbol> *neurons,
	map <string, Symbol> *weight,
	int filter,
	Shape kernel,
	Shape stride,
	Shape pad,
	bool isRelu,
	bool noBias){

	string sw = string("w") + name;
	
	string sb = string("b") + name;

	string sr = string("r") + name;

	string in = string(input);

	string out = string(name);

	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*weight)[sb] = Symbol::Variable(sb);

	}

	if(isRelu)

		(*neurons)[out] = relu(sr, Convolution(out, (*neurons)[in], (*weight)[sw], (*weight)[sb], kernel, filter, stride, Shape(1,1), pad));

	else

		(*neurons)[out] = relu(Convolution(out, (*neurons)[in], (*weight)[sw], (*weight)[sb], kernel, filter, stride, Shape(1,1), pad));


}

inline Symbol flt::mx::layer::conv(string snode,
	Symbol &input,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	map <string, Symbol> *aux,
	int filter,
	string act,
	bool isTraining,
	Shape kernel,
	Shape stride,
	Shape pad,
	bool isRelu,
	bool noBias){


	string sw = string("w") + snode;
	
	string sb = string("b") + snode;

	string sr = string("r") + snode;
	string sgamma = string("gamma") + snode;

	string sbeta = string("beta") + snode;

	string smean = string("mean") + snode;

	string svar = string("var") + snode;


	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

		if (act == "bn"){

			(*weight)[sgamma] = Symbol::Variable(sgamma);

			(*weight)[sbeta] = Symbol::Variable(sbeta);

			(*aux)[smean] = Symbol::Variable(smean);

			(*aux)[svar] = Symbol::Variable(svar);

		}

	}
	
	Symbol node = Convolution(snode, input, (*weight)[sw], (*bias)[sb], kernel, filter, stride, Shape(1,1), pad);

	if (act == "bn")

		node = BatchNorm(node, (*weight)[sgamma], (*weight)[sbeta], (*aux)[smean], (*aux)[svar], 0.001, 0.9, 1, false);
	
	if(isRelu)

		return LeakyReLU(sr, node);

	else

		return node;
}
inline Symbol flt::mx::layer::conv(string snode,
	Symbol &input,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	int filter,
	Shape kernel,
	Shape stride,
	Shape pad,
	bool isRelu,
	bool noBias){

	string sw = string("w") + snode;
	
	string sb = string("b") + snode;

	string sr = string("r") + snode;
	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

	}

	if(isRelu)

		return LeakyReLU(sr, Convolution(snode, input, (*weight)[sw], (*bias)[sb], kernel, filter, stride, Shape(1,1), pad));

	else

		return Convolution(snode, input, (*weight)[sw], (*bias)[sb], kernel, filter, stride, Shape(1,1), pad);

}
inline void flt::mx::layer::deconv(char * prefix, 
	char * name,
	char * input,
	map <string, Symbol> *neurons,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	int filter,
	Shape kernel,
	Shape stride,
	Shape pad,
	Shape adj,
	Shape dilate,
	bool isRelu,
	bool noBias){
	
	string sw = string("w") + name;
	
	string sb = string("b") + name;
	
	string sr = string("r") + name;
	
	string in = string(prefix) + input;

	string out = string(prefix) + name;

	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);
	}

	Symbol node = Deconvolution(out, (*neurons)[in], (*weight)[sw], (*bias)[sb], kernel, filter, stride, dilate, pad, adj, Shape(), 1, 512, false);

	if(isRelu)

		(*neurons)[out] = relu(sr, node);

	else

		(*neurons)[out] = node;

};
 
inline Symbol flt::mx::layer::deconv(string snode,
	Symbol &input,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	int filter,
	Shape kernel,
	Shape stride,
	Shape pad,
	Shape adj,
	Shape dilate,
	bool isRelu,
	bool noBias){
	
	string sw = string("w") + snode;
	
	string sb = string("b") + snode;
	
	string sr = string("r") + snode;

	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

	}

	Symbol node = Deconvolution(snode, input, (*weight)[sw], (*bias)[sb], kernel, filter, stride, dilate, pad, adj, Shape(), 1, 512, false);
	

	if(isRelu)
		
		return LeakyReLU(sr, node);
		//return relu(sr, node);

	else

		return node;

};
inline Symbol flt::mx::layer::deconv(string snode,
	Symbol &input,
	map <string, Symbol> *weight,
	map <string, Symbol> *bias,
	map <string, Symbol> *aux,
	int filter,
	string act,
	bool isTraining,
	Shape kernel,
	Shape stride,
	Shape pad,
	Shape adj,
	Shape dilate,
	bool isRelu,
	bool noBias){
	
	string sw = string("w") + snode;
	
	string sb = string("b") + snode;
	
	string sr = string("r") + snode;

	string sgamma = string("gamma") + snode;

	string sbeta = string("beta") + snode;

	string smean = string("mean") + snode;

	string svar = string("var") + snode;

	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

		if (act == "bn"){

			(*weight)[sgamma] = Symbol::Variable(sgamma);

			(*weight)[sbeta] = Symbol::Variable(sbeta);

			(*aux)[smean] = Symbol::Variable(smean);

			(*aux)[svar] = Symbol::Variable(svar);

		}

	}

	Symbol node = Deconvolution(snode,
			input,
			(*weight)[sw],
			(*bias)[sb],
			kernel,
			filter,
			stride,
			dilate,
			pad,
			adj,
			Shape(),
			1,
			512,
			false);
	

	if(act == "bn")
		
		node = BatchNorm(node, (*weight)[sgamma], (*weight)[sbeta], (*aux)[smean], (*aux)[svar], 0.001, 0.9, 1, false);

	if(isRelu)
		
		return LeakyReLU(sr, node);
		//return sigmoid(node);

	else

		return node;

};
inline void flt::mx::layer::fullyconnected(char * prefix, 
		char * name,
		char * input,
		map <string, Symbol> *neurons,
		map <string, Symbol> *weight,
		map <string, Symbol> *bias,
		int nout){
	
	string sw = string("w") + name;

	string sb = string("b") + name;
	
	string in = string(prefix) + input;
	
	string out = string(prefix) + name;

	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

	}
	
	(*neurons)[out] = FullyConnected(out, (*neurons)[in], (*weight)[sw], (*bias)[sb], nout);

}

inline Symbol flt::mx::layer::fullyconnected(string snode,
		Symbol & input,
		map <string, Symbol> *weight,
		map <string, Symbol> *bias,
		int nout){
	
	string sw = string("w_") + snode;

	string sb = string("b_") + snode;
	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

	}
	
	return FullyConnected(snode, input, (*weight)[sw], (*bias)[sb], nout);

}

inline Symbol flt::mx::layer::fullyconnected(string snode,
		Symbol & input,
		map <string, Symbol> *weight,
		map <string, Symbol> *bias,
		map <string, Symbol> *aux,
		string act,
		bool isTraining,
		int nout){
	
	string sw = string("w_") + snode;

	string sb = string("b_") + snode;
	
	string sgamma = string("gamma") + snode;

	string sbeta = string("beta") + snode;

	string smean = string("mean") + snode;

	string svar = string("var") + snode;
	
	if((*weight).find(sw) == (*weight).end()){

		(*weight)[sw] = Symbol::Variable(sw);

		(*bias)[sb] = Symbol::Variable(sb);

		if (act == "bn"){

			(*weight)[sgamma] = Symbol::Variable(sgamma);

			(*weight)[sbeta] = Symbol::Variable(sbeta);

			(*aux)[smean] = Symbol::Variable(smean);

			(*aux)[svar] = Symbol::Variable(svar);

		}

	}


	Symbol node = FullyConnected(snode, input, (*weight)[sw], (*bias)[sb], nout);

	if (act == "bn")

		return BatchNorm(node, (*weight)[sgamma], (*weight)[sbeta], (*aux)[smean], (*aux)[svar], 0.001, 0.9, 1, false);
	
	else

		return node;
}
inline void flt::mx::layer::maxpool(char * output,
		char * input,
		map <string, Symbol> *neurons,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
		
	string out = string(output);

	string in = string(input);

	(*neurons)[out] = Pooling(out, (*neurons)[in], kernel, PoolingPoolType::kMax, false, false, method, stride, pad);

}
inline void flt::mx::layer::maxpool(char * p,
		char * output,
		char * input,
		map <string, Symbol> *neurons,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
		
	string out = string(p) + output;

	string in = string(p) + input;

	(*neurons)[out] = Pooling(out, (*neurons)[in], kernel, PoolingPoolType::kMax, false, false, method, stride, pad);

}

inline Symbol flt::mx::layer::maxpool(string snode,
		Symbol &input,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
		
	return  Pooling(snode, input, kernel, PoolingPoolType::kMax, false, false, method, stride, pad);

}

inline void flt::mx::layer::avgpool(char * p, 
		char * output,
		char * input,
		map <string, Symbol> *neurons,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
	
	string out = string(p) + output;

	string in = string(p) + input;

	(*neurons)[out] = Pooling(out, (*neurons)[in], kernel, PoolingPoolType::kAvg, false, false, method, stride, pad);
}
inline void flt::mx::layer::avgpool(char * output,
		char * input,
		map <string, Symbol> *neurons,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
	
	string out = string(output);

	string in = string(input);

	(*neurons)[out] = Pooling(out, (*neurons)[in], kernel, PoolingPoolType::kAvg, false, false, method, stride, pad);
}


inline Symbol flt::mx::layer::avgpool(string snode,
		Symbol &input,
		Shape kernel,
		Shape stride,
		Shape pad,
		PoolingPoolingConvention method){
	
	return Pooling(snode, input, kernel, PoolingPoolType::kAvg, false, false, method, stride, pad);
}
inline void flt::mx::layer::concat(char * p,
		char * output,
		vector <Symbol> *v,
		map <string, Symbol> *neurons,
		int dimension){
	
	string out = string(p) + output;

	(*neurons)[out] = Concat(out, (*v), (*v).size(), dimension);
}

inline Symbol flt::mx::layer::concat(string snode, vector <Symbol> *v, int dimension){
	return Concat(snode, (*v), (*v).size(), dimension);
}
inline Symbol flt::mx::layer::concat(string snode, vector <Symbol> &&v, int dimension){

	return Concat(snode, (v), (v).size(), dimension);
}
inline Symbol flt::mx::layer::concat(char * p, char * output, vector <Symbol> *v, int dimension){
	
	string out = string(p) + output;
	
	return Concat(out, (*v), (*v).size(), dimension);
}

#endif
