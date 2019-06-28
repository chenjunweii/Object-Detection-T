#ifndef FLT_STL_HH
#define FLT_STL_HH

#include <iostream>
#include <vector>
#include "src/stl.h"

using namespace std;


inline vector <int> flt::fvector::slice(vector <int> v, int begin, int end){

	vector <int> sliced;

	for(int i = begin; i != end; i++)

		sliced.emplace_back(v[i]);

	return sliced;
}


inline vector <int> flt::fvector::concat(vector <int> x, vector <int> y){

	for(int i = 0; i != y.size(); i++)

		x.emplace_back(y[i]);

	return x;

}

inline vector <float> flt::fvector::arange(float vmin, float vmax, float step){

	vector <float> v;

	while(vmin < vmax){

		v.emplace_back(vmin);

		vmin += step;
	}

	return v;
}

inline void flt::fvector::print(vector <string> &v, bool p = false){

	for(vector <string>::iterator i = v.begin(); i != v.end(); i++){

		cout << (*i) << endl;

	}

	if(p){

		cout << endl << "total : " << v.size() << endl;

		cout << endl << "======================================================" << endl;

	}
}

#endif
