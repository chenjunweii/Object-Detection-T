#ifndef FLT_FILE_HH
#define FLT_FILE_HH
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

#include "src/file.h"

using namespace std;

flt::ffile::fiterator::fiterator(std::string _filename){

	filename = _filename;

	line_number = 0;

	if (!boost::filesystem::exists(_filename))

		cout << "Iterator list is not exist" << endl;

	f = ifstream(_filename, ifstream::in);
}


inline int flt::ffile::fiterator::next(){

	if (f >> line){

		line_number += 1;

		return 1;
	}

	else

		return 0;

}

#endif

