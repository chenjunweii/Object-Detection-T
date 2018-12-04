#include <iostream>
#include <fstream>
#include "executor.h"
#include <numeric>


using namespace tvm::runtime;
using namespace std;


TVMExecutor::TVMExecutor(string network, map <string,
		vector <int64_t>> & inshapes,
		map <string, vector <int64_t>> & outshapes,
		vector <string> & _outnode,
		string device,
		bool use_params = true,
		string dtype = "float32", int device_id = 0) : device_id(device_id), output(_outnode) {

	string sso = string("./so/") + network + ".tvm.so";

	string sjson = string("./graph/") + network + ".tvm.json";

	string sparams = string("./params/") + network + ".tvm.params";

	if (device.compare("gpu") == 0)

		device_type = kDLGPU;

	else

		device_type = kDLCPU;

	if (dtype.compare("float32") == 0){

		dtype_code = kDLFloat;
		
		dtype_bits = 32;
		
		dtype_lanes = 1;

		dtype_bytes = 4;
	}

	else 

        throw invalid_argument("[!] Data Type is not support yet");

	Module so = Module::LoadFromFile(sso);

    ifstream json_stream(sjson, ios::in);

    string json_data((istreambuf_iterator <char> (json_stream)), istreambuf_iterator <char> ());
    
	json_stream.close();

	module = (*Registry::Get("tvm.graph_runtime.create"))(json_data, so, device_type, device_id);
	
	if (use_params){
		
		ifstream params_stream(sparams, ios::binary);
    
		string params_data((istreambuf_iterator <char> (params_stream)), istreambuf_iterator <char> ());
		
		params_stream.close();

		params.data = params_data.c_str();
		
		params.size = params_data.length();

		module_load_params = module.GetFunction("load_params");
		
		module_load_params(params);
	}

	execute = module.GetFunction("run");

	module_set_input = module.GetFunction("set_input");
    
	module_get_output = module.GetFunction("get_output");

	for (auto & s : inshapes){

		nds[s.first] = nullptr;

		//cout << "after pipe " << endl;

		//int64_t shape[4] = {1, 3, 512, 512};//[s.second.size()] = int64_t[s.second.size()];
		//int64_t shape[4] = {1, 3, 512, 512};//[s.second.size()] = int64_t[s.second.size()];a

		TVMArrayAlloc(s.second.data(), s.second.size(), dtype_code, dtype_bits, dtype_lanes, device_type, device_id, & nds[s.first]);
		//TVMArrayAlloc(shape, s.second.size(), dtype_code, dtype_bits, dtype_lanes, device_type, device_id, & nds[s.first]);

		pipes[s.first] = TVMPipe(s.second, device, dtype, device_id);

		//auto fsize = accumulate(begin(s.second), end(s.second), 1, std::multiplies <int64_t> () );

		//fs[s.first] = vector <float> (fsize);

		module_set_input(s.first, nds[s.first]);

	}

	

	for (auto & o : outshapes){

		nds[o.first] = nullptr;

		bool uncertain = false;

		for (auto & _s : o.second)

			if (_s == -1)

				uncertain = true;

		if (uncertain){

			pipes[o.first] = TVMPipe(o.second, device, dtype, device_id);
			
		}

		else{
		
			TVMArrayAlloc(o.second.data(), o.second.size(), dtype_code, dtype_bits, dtype_lanes, device_type, device_id, & nds[o.first]);
			
			pipes[o.first] = TVMPipe(o.second, device, dtype, device_id);
			
			auto fsize = accumulate(begin(o.second), end(o.second), 1, multiplies <int64_t> () );
			
			fs[o.first] = vector <float> (fsize);
		
		}

	}

	for (int i = 0; i != output.size(); ++i){
		
		output_index[output[i]] = i;
	}

	execute();

}

void TVMExecutor::Load(string node, Mat & in){

	pipes[node].MatToTVMArray(in, nds[node]);

	module_set_input(node, nds[node]);

}

void TVMExecutor::Load(string node, DLTensor * in, bool copy = false){

	//pipes[node].MatToTVMArray(in, nds[node]);
	//
	//
	int status = 0;
	
	if (copy){
	
		status = TVMArrayCopyFromTo(in, nds[node], NULL);
	}

	else{
		
		nds[node] = in;
	}

	module_set_input(node, nds[node]);

	if (status < 0)

		throw runtime_error("[!] Load :: TVMArrayCopyFromTo Error ...");

}

void TVMExecutor::Get(string node, vector <float> & out){

	module_get_output(output_index[node], nds[node]);

	pipes[node].TVMArrayToFloatArray(nds[node], out);

}


void TVMExecutor::GetOutput(){

	for (auto & o : output){

		Get(o, fs[o]);		
	}
}


TVMExecutor::~TVMExecutor(){

	for (auto & nd : nds)

		TVMArrayFree(nd.second);
}


void TVMExecutor::Forward(){

	execute();
	
}


void TVMExecutor::Show(string node){



}
