#ifndef FLT_MX_SHAPE_H
#define FLT_MX_SHAPE_H

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>

using namespace std;
using namespace mxnet::cpp;

namespace flt{
	
	namespace mx{
		
		inline void print_nd_shape(map <string, NDArray> &m, bool p);

		/* print map */

		inline void print_uint_shape(map <string, vector <mx_uint>> &shape_map);
		
		/* print vector */

		inline void print_uint_shape(vector <vector <mx_uint>> &shape_vector, bool print_total = true);
		
		/* print shape */

		inline void print_uint_shape(vector <mx_uint> &shape_vector);

		inline void print_arch(map <string, Symbol> &neurons, map <string, vector <mx_uint>> &arg_shapes);

		inline Shape get_shape(Symbol in, map <string, vector <mx_uint>> &arg_shapes);

		inline void infer_partial(Symbol symbol,
			map <string, vector <mx_uint>> &arg_shape,
			 vector <vector <mx_uint>> &input,
			  vector <vector <mx_uint>> &aux,
			   vector <vector <mx_uint>> &out);

		inline int total(Shape s);

		inline map <string, vector <mx_uint>> concat_uint_map(map <string, vector <mx_uint>> &m1, map <string, vector <mx_uint>> &m2);

		inline map <string, Shape> merge_uint_vector(vector <string> &vname, vector <vector <mx_uint>> &vshape);
	
	} /* mx */

} /* flt */
#endif
