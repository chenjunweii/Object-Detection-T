#ifndef FLT_DEBUG_HH
#define FLT_DEBUG_HH

#include <iostream>
#include "src/debug.h"

using namespace std;

inline int flt::fdebug::error(string s){

	cout << "[!] Error : " << s << endl;
}

inline int flt::fdebug::log(string s, bool b){

	if(b)

		cout << "[*] Log : " << s << endl;
}

#endif
