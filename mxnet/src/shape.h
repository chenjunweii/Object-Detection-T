#ifndef FLT_SHAPE_H
#define FLT_SHAPE_H

#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <random>

using namespace std;
using namespace mxnet::cpp;

namespace flt{	

	class fshape{	
		
		public:

			int rank;
			
			int size = 1;
			
			vector <int> v;
		
			int operator [] (int i);
			
			friend ostream & operator << (ostream & stream, fshape const & instance){
				
				cout << "[";
				
				for(int i = 0; i != instance.rank; ++i){

					stream << instance.v[i];

					if(i != instance.rank - 1)

						cout << " ";
				}
			

				cout << "]";

				stream << endl;

				return stream;
			}

			//fshape * operator = (fshape & instance);	
			fshape & operator = (fshape & instance);	
			
			
			template <typename T, typename... Args>

			inline fshape(T t, Args... args);
			
			template <typename T>

			inline fshape(T t);

			template <typename T>

			inline void construct(T t);

			template <typename T, typename... Args>
			
			inline void construct(T t, Args... args);

	}; /* fshape */

} /* flt */

#endif
