#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "BatchStreamPPM.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "common.h"
#include "src/plugin.hh"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

//static constexpr int OUTPUT_CLS_SIZE = 91;
static constexpr int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const char* OUTPUT_BLOB_NAME0 = "NMS";
static Logger gLogger;
static samples_common::Args args;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "sample_uff_ssd: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)


ICudaEngine* loadModelAndCreateEngine(const char * uffFile, int maxBatchSize,
                                      IUffParser * parser, nvuffparser::IPluginFactory * pluginFactory,
                                      IHostMemory *& trtModelStream)
{
    // Create the builder
    IBuilder * builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition * network = builder->createNetwork();

    parser->setPluginFactory(pluginFactory);

    std::cout << "Begin parsing model..." << std::endl;

    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
	builder->setFp16Mode(true);

    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    std::cout << "End building engine..." << std::endl;

    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();

    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
}

static constexpr int CAL_BATCH_SIZE = 50;

static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

int main(int argc, char* argv[]){
	
	cout << "DawdwA" << endl;

   	string uff = "mobilenet_v2_custom.pb.uff";

    const int N = 1;

    auto parser = createUffParser();
    
	parser->registerInput("Input", DimsCHW(3, 300, 300), UffInputOrder::kNCHW);
    
	parser->registerOutput("MarkOutput_0");
    
	IHostMemory * trtModelStream = nullptr;
    
	PluginFactory pluginFactorySerialize;
    
	//BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

    //Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, "CalibrationTableSSD");

	cout << "load Model " << endl;
    
	ICudaEngine* tmpEngine = loadModelAndCreateEngine(uff.c_str(), N, parser, &pluginFactorySerialize, trtModelStream);
    
	assert(tmpEngine != nullptr);
    
	assert(trtModelStream != nullptr);

	ofstream ofs("mobilenet_v2_custom.trt", ios::out | ios::binary);

	ofs.write((char *) (trtModelStream->data()), trtModelStream->size());

	ofs.close();
    
	tmpEngine->destroy();
	
	pluginFactorySerialize.destroyPlugin();

}
	/*


    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

    parser->registerInput("Input", DimsCHW(3, 300, 300), UffInputOrder::kNCHW);
    parser->registerOutput("MarkOutput_0");

    IHostMemory* trtModelStream{nullptr};

    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, "CalibrationTableSSD");

    PluginFactory pluginFactorySerialize;
    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, &pluginFactorySerialize, &calibrator, trtModelStream);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();
    pluginFactorySerialize.destroyPlugin();

}



*/
