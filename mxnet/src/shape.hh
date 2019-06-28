#ifndef FLT_MX_SHAPE_HH
#define FLT_MX_SHAPE_HH

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <map>
#include "src/mx/shape.h"
#include "src/debug.h"

using namespace std;
using namespace mxnet::cpp;

typedef vector <mx_uint> mx_shape;
	
		
inline void flt::mx::print_nd_shape(map <string, NDArray> &m, bool p = false){

	for(map <string, NDArray> :: iterator i = m.begin(); i != m.end(); i++){

		cout << (*i).first << " : " << Shape((*i).second.GetShape()) << endl;
	}

	if(p){

		cout << endl << "total : " << m.size() << endl;

		cout << endl << "======================================================" << endl;

	}
}


inline void flt::mx::print_uint_shape(vector <vector <mx_uint>> &v, bool print_total){

	for(vector <vector <mx_uint>> :: iterator i = v.begin(); i != v.end(); i++)

		cout << Shape((*i)) << endl;

	if(print_total){

		cout << endl << "total : " << v.size() << endl;

		cout << endl << "======================================================" << endl;

	}
}

/*
 
inline void flt::mx::print_uint_shape(vector <vector <mx_uint>> &shape_vector){

	if(!shape_vector.size())

		cout << "[!] No items in Vector " << endl;

	else

		for(int i = 0; i != shape_vector.size(); i++){

			cout << "[";

			for(int j = 0; j != shape_vector[i].size(); j++){

				cout << shape_vector[i][j];

				if(j != shape_vector[i].size() - 1)

					cout << ", ";
			}

			cout << "]" << endl;

		}

}

*/

inline map <string, vector <mx_uint>> flt::mx::concat_uint_map(map <string, vector <mx_uint>> &m1, map <string, vector <mx_uint>> &m2){

	map <string, vector <mx_uint>> m3;

	for(map <string, vector <mx_uint>> :: iterator i = m1.begin(); i != m1.end(); i++)

		m3[(*i).first] = (*i).second;

	for(map <string, vector <mx_uint>> :: iterator i = m2.begin(); i != m2.end(); i++)

		m3[(*i).first] = (*i).second;

	return m3;
}

inline map <string, Shape> flt::mx::merge_uint_vector(vector <string> &vname, vector <vector <mx_uint>> &vshape){

	map <string, Shape> shape_map;

	if (vname.size() != vshape.size())

		flt::fdebug::error("size of two vectors are diffenet, can't merge to a map");

	else{

			typedef vector <string>::iterator iname;
			typedef vector <vector <mx_uint>>::iterator ishape ;

			for(pair <iname, ishape> i(vname.begin(), vshape.begin()); i.first != vname.end() && i.second != vshape.end(); ++i.first, ++i.second){

				shape_map[(*i.first)] = Shape(*i.second);

			}

		}

	return shape_map;

}




inline void flt::mx::print_uint_shape(map <string, vector <mx_uint>> &shape_map){

	if(!shape_map.size())

		cout << "[!] No items in Vector " << endl;

	else

		for(map <string, vector <mx_uint>>::iterator it = shape_map.begin();

			it != shape_map.end(); it++){

			cout << it->first << " : [";

			for(int i = 0; i != it->second.size(); i++){

				cout << it->second[i];

				if(i != it->second.size() - 1)

					cout << ", ";

			}

			cout << "]"<< endl;

		}
}



inline void flt::mx::print_uint_shape(vector <mx_uint> &shape_vector){

	cout << "[";

	for(int i = 0; i != shape_vector.size(); i++){

		cout << shape_vector[i];

		if(i != shape_vector.size() - 1)

			cout << ", ";

	}

	cout << "]" << endl;
}


inline Shape flt::mx::get_shape(Symbol in, map <string, vector <mx_uint>> &arg_shape){

	vector <vector <mx_uint>> input;
	vector <vector <mx_uint>> aux;
	vector <vector <mx_uint>> out;
	vector <mx_uint> ret;

	//cout << "get 0 ..." << endl;

	map <string, vector <mx_uint>> partial_shape;

	//cout << "get ... " << endl;

	//cout << "arg : " << Shape(arg[string("inputs")]) << endl;

	vector <string> partial_arg = in.ListArguments();

	//cout << "Partial arg name : ";

	for(vector <string> :: iterator i = partial_arg.begin(); i != partial_arg.end(); ++i){

		//cout << (*i) << " : "<< Shape(arg_shape[(*i)]) << endl;

		partial_shape[(*i)] = arg_shape[(*i)];

	}
	//cout << "Get 1..." << endl;

	in.InferShape(partial_shape, &input, &aux, &out);

	//cout << "get 2 ... " << endl;

	for(int i = 0; i != out.size(); i++){

		for(int j = 0; j != out[i].size(); j++)

			ret.push_back(out[i][j]);
	}

	return Shape(ret);
}

inline int flt::mx::total(Shape s){


}

inline void flt::mx::infer_partial(Symbol symbol,
	map <string, vector <mx_uint>> &arg_shape,
	 vector <vector <mx_uint>> &input,
	  vector <vector <mx_uint>> &aux,
	   vector <vector <mx_uint>> &out){

	map <string, vector <mx_uint>> partial_shape;

	vector <string> partial_arg = symbol.ListArguments();

	for(vector <string> :: iterator i = partial_arg.begin(); i != partial_arg.end(); ++i)

		partial_shape[(*i)] = arg_shape[(*i)];

	//cout << "this is infer partial_shape ... " << endl;

	symbol.InferShape(partial_shape, &input, &aux, &out);

	//	cout << "this is infer partial_shape ... " << endl;
}


inline void flt::mx::print_arch(map <string, Symbol> &neurons, map <string, vector <mx_uint>> &arg_shapes){

	cout << endl << "================ Architecture ================" << endl << endl;

	for(map <string, Symbol> ::iterator it = neurons.begin(); it != neurons.end(); it++){

		//cout << (*it).first << endl;

		cout << (*it).first << " : " << get_shape((*it).second, arg_shapes) << endl;

	}

	cout << endl << "================ Architecture ================" << endl << endl;
}

#endif
