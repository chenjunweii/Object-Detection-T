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

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

static Logger gLogger;
static samples_common::Args args;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "sample_uff_ssd: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)

static constexpr int OUTPUT_CLS_SIZE = 91;
static constexpr int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const char* OUTPUT_BLOB_NAME0 = "NMS";

//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

// Concat layers
// mbox_priorbox, mbox_loc, mbox_conf
const int concatAxis[2] = {1, 1};
const bool ignoreBatch[2] = {false, false};

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 200, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};

// Visualization
const float visualizeThreshold = 0.5;

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(samples_common::getElementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * samples_common::getElementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpyAsync(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs, std::max_element(outputs, outputs + eltCount));

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/ssd/",
                                  "data/ssd/VOC2007/",
                                  "data/ssd/VOC2007/PPMImages/",
                                  "data/samples/ssd/",
                                  "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

void populateTFInputData(float* data)
{

    auto fileName = locateFile("inp_bus.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        istringstream iss(line);
        float num;
        iss >> num;
        data[id++] = num;
    }

    return;
}

void populateClassLabels(std::string (&CLASSES)[OUTPUT_CLS_SIZE])
{

    auto fileName = locateFile("ssd_coco_labels.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        CLASSES[id++] = line;
    }

    return;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samples_common::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, nvuffparser::IPluginFactory* pluginFactory,
                                      IInt8Calibrator* calibrator, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();
    parser->setPluginFactory(pluginFactory);

    std::cout << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    builder->setHalf2Mode(false);
    if (args.runInInt8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

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

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();

    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samples_common::safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto t_start = std::chrono::high_resolution_clock::now();
    context.execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
#ifdef SSD_INT8_DEBUG
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                    buffers[bindingIdx]);
#endif
    }

    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}


int main(int argc, char* argv[])
{
    // Parse command-line arguments.

    // Read a random sample image.
    srand(unsigned(time(nullptr)));
    // Available images.
    std::vector<std::string> imageList = {"dog.ppm"};
    std::vector<samples_common::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(N);

    assert(ppms.size() <= imageList.size());
    std::cout << " Num batches  " << N << std::endl;
    for (int i = 0; i < N; ++i)
    {
        readPPMFile(imageList[i], ppms[i]);
    }

    vector<float> data(N * INPUT_C * INPUT_H * INPUT_W);

    for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
    {
        for (int c = 0; c < INPUT_C; ++c)
        {
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j) {
                data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * INPUT_C + c]) - 1.0;
            }
        }
    }
    std::cout << " Data Size  " << data.size() << std::endl;

    // Deserialize the engine.
    std::cout << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Host memory for outputs.
    vector<float> detectionOut(N * detectionOutputParam.keepTopK * 7);
    vector<int> keepCount(N);

    // Run inference.
    doInference(*context, &data[0], &detectionOut[0], &keepCount[0], N);
    cout << " KeepCount " << keepCount[0] << "\n";

    std::string CLASSES[OUTPUT_CLS_SIZE];

    populateClassLabels(CLASSES);

    for (int p = 0; p < N; ++p)
    {
        for (int i = 0; i < keepCount[p]; ++i)
        {
            float* det = &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
            if (det[2] < visualizeThreshold) continue;

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            assert((int) det[1] < OUTPUT_CLS_SIZE);
            std::string storeName = CLASSES[(int) det[1]] + "-" + std::to_string(det[2]) + ".ppm";

            printf("Detected %s in the image %d (%s) with confidence %f%% and coordinates (%f,%f),(%f,%f).\nResult stored in %s.\n", CLASSES[(int) det[1]].c_str(), int(det[0]), ppms[p].fileName.c_str(), det[2] * 100.f, det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H, storeName.c_str());

            samples_common::writePPMFileWithBBox(storeName, ppms[p], {det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H});
        }
    }

    // Destroy the engine.
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Destroy plugins created by factory
    pluginFactory.destroyPlugin();

    return EXIT_SUCCESS;
}
