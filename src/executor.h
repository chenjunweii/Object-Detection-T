#ifndef FLT_EXECUTOR_H
#define FLT_EXECUTOR_H


#include "tvm.h"
#include "pipe.h"

using namespace std;
using namespace tvm::runtime;


class TVMExecutor {

	public:

		inline TVMExecutor(string network,
				map <string, vector <int64_t>> & inshapes,
				map <string, vector <int64_t>> & outshapes,
				vector <string> & outnodes, string device, bool use_params, string dtype, int device_id);
		
		inline void Load(string node, Mat & in);

		inline void Load(string node, DLTensor * in, bool copy);

		inline void Get(string node, vector <float> & out);

		inline void Forward();

		inline void GetOutput();
		//inline execute();

		TVMByteArray params;

		inline ~ TVMExecutor();

		int dtype_code, dtype_bits, dtype_lanes, dtype_bytes, device_type, device_id;
		
		Module module;
	
		PackedFunc execute, module_set_input, module_load_params, module_get_output;

		map <string, DLTensor*> nds;

		map <string, vector <float>> fs;

		map <string, TVMPipe> pipes;

		vector <string> output;

		map <string, int> output_index;

		inline void Show(string node);

};


#endif
