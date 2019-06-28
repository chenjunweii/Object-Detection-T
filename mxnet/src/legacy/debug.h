#ifndef FLT_DEBUG_H
#define FLT_DEBUG_H

#include <iostream>


using namespace std;

namespace flt{

	namespace fdebug{
		
		inline int error(string); // print error

		inline int log(string, bool); // debug log, print if debug flag is set to True

	}
}


#endif
