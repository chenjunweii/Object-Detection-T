#ifndef FLT_FILE_H
#define FLT_FILE_H

#include <iostream>
#include <fstream>

using namespace std;

namespace flt{

	namespace ffile{

		class fiterator{

			public:

				/* variable */

				std::string line;

				std::string filename;

				std::ifstream f;

				int line_number;

				/* function */

				fiterator(std::string _filename);
				
				int next();

		};

	} /* ffile */

} /* flt */


#endif
