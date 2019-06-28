#ifndef FLT_HDF5_H
#define FLT_HDF5_H

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <H5Cpp.h>  
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace H5;
using namespace mxnet::cpp;

namespace flt{	

	class fhdf5{

			const char * filename;

		
		public:
				
			Context *ctx;
			
			H5File file;

			fhdf5 ();
		
			fhdf5 (const char * f);

			fhdf5 (const char * f, Context *ctx);
		
			fhdf5 (string f);
			
			~ fhdf5 ();

			inline void read();

			inline void open();
			
			inline void open_exist();
			
			inline void close();
			
			inline void create();
			
			inline void list_object();

			inline void save(map <string, vector <float>> &m);

			inline void save_NDArray(map <string, NDArray> &ndarg);
		
			inline void load_keras_all();
			
			inline void load_keras(string node);
			
			inline void load_weight();

			/*  keras use vector <float> instead of NDArray, so the float array we dont need will not load to Context
			 *
			 *	we call choose which float array to load to Context manually
			 * 
			 * 	but we load all the weight to nddata when we are using trained weight to predict result
			 *
			 * */
			
			map <string, map <string, vector <float>>> keras;  // keras has two dataset in each layer(Group)
			
			map <string, NDArray> nddata; // for our own purpose
			
			map <string, vector <float>> fdata;


		private:

			static herr_t iterate_all_object(hid_t loc_id, const char * name, const H5O_info_t * linfo, void * data);

			static herr_t iterate_keras_group(hid_t loc_id, const char * name, const H5L_info_t * linfo, void * data);
			
			static herr_t iterate_weight_object(hid_t loc_id, const char * name, const H5O_info_t * linfo, void * data);

	}; /* fhdf5*/

} /* fhdf5 */


#endif
