#ifndef FLT_TRT_LOADER
#define FLT_TRT_LOADER

#include "base.h"
#include "plugin.hh"

//using namespace std;

namespace flt{

void load_serialized_model(string filename, char ** data,  IRuntime ** runtime, ICudaEngine ** engine, IExecutionContext ** context, PluginFactory * plugin){

	ifstream ifs(filename, ios::in | ios::binary);

	ifs.seekg (0, ifs.end);
    
	int size = ifs.tellg();
    
	ifs.seekg (0, ifs.beg);

	*data = new char [size];

	ifs.read(*data, size); 

	ifs.close();

	(*runtime) = createInferRuntime(logger);

	(*engine) = (*runtime)->deserializeCudaEngine(*data, size, plugin);
    
	(*context) = (*engine)->createExecutionContext();
	
}


} // flt


#endif
