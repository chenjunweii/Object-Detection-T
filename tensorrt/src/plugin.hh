#ifndef FLT_TRT_PLUGIN
#define FLT_TRT_PLUGIN
#include "base.h"

static constexpr int OUTPUT_CLS_SIZE = 91;
const int concatAxis[2] = {1, 1};
const bool ignoreBatch[2] = {false, false};
DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 200, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};
class FlattenConcat : public IPlugin
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
    {
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    }

    FlattenConcat(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        mIgnoreBatch = read<bool>(d);
        mConcatAxisID = read<int>(d);
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
        mOutputConcatAxis = read<int>(d);
        mNumInputs = read<int>(d);
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

        std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

        mCHW = read<nvinfer1::DimsCHW>(d);

        std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

        assert(d == a + length);
    }
    ~FlattenConcat()
    {
        CHECK(cudaFreeHost(mInputConcatAxis));
        CHECK(cudaFreeHost(mCopySize));
    }
    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims >= 1);
        assert(index == 0);
        mNumInputs = nbInputDims;
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        mOutputConcatAxis = 0;
#ifdef SSD_INT8_DEBUG
        std::cout << " Concat nbInputs " << nbInputDims << "\n";
        std::cout << " Concat axis " << mConcatAxisID << "\n";
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 3; ++j)
                std::cout << " Concat InputDims[" << i << "]"
                          << "d[" << j << " is " << inputs[i].d[j] << "\n";
#endif
        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            assert(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1) assert(inputs[i].d[0] == inputs[0].d[0]);
            if (mConcatAxisID != 2) assert(inputs[i].d[1] == inputs[0].d[1]);
            if (mConcatAxisID != 3) assert(inputs[i].d[2] == inputs[0].d[2]);
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 3 ? mOutputConcatAxis : 1);
    }

    int initialize() override
    {
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int numConcats = 1;
        assert(mConcatAxisID != 0);
        numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

        if (!mIgnoreBatch) numConcats *= batchSize;

        float* output = reinterpret_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            float* inputTemp;
            CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

            CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

            for (int n = 0; n < numConcats; ++n)
            {
                CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
            }
            CHECK(cudaFree(inputTemp));
            offset += mInputConcatAxis[i];
        }

        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
    }

    void serialize(void* buffer) override
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, mIgnoreBatch);
        write(d, mConcatAxisID);
        write(d, mOutputConcatAxis);
        write(d, mNumInputs);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mInputConcatAxis[i]);
        }
        write(d, mCHW);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mCopySize[i]);
        }
        assert(d == a + getSerializationSize());
    }

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        assert(nbOutputs == 1);
        mCHW = inputs[0];
        assert(inputs[0].nbDims == 3);
        CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
        }
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize;
    bool mIgnoreBatch{false};
    int mConcatAxisID, mOutputConcatAxis, mNumInputs;
    int* mInputConcatAxis;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
};

// Integration for serialization.
class PluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory
{
public:
    std::unordered_map<std::string, int> concatIDs = {
        std::make_pair("_concat_box_loc", 0),
        std::make_pair("_concat_box_conf", 1)};

        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const nvuffparser::FieldCollection fc) override
        {
            assert(isPlugin(layerName));

            const nvuffparser::FieldMap* fields = fc.fields;
            int nbFields = fc.nbFields;

            if(!strcmp(layerName, "_PriorBox"))
            {
                assert(mPluginPriorBox == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                float minScale = 0.2, maxScale = 0.95;
                int numLayers;
                std::vector<float> aspectRatios;
                std::vector<int> fMapShapes;
                std::vector<float> layerVariances;

                for(int i = 0; i < nbFields; i++)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "numLayers") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        numLayers = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "minScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        minScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        maxScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "aspectRatios")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        aspectRatios.reserve(size);
                        const double *aR = static_cast<const double*>(fields[i].data);
                        for(int j=0; j < size; j++)
                        {
                            aspectRatios.push_back(*aR);
                            aR++;
                        }
                    }
                    else if(strcmp(attr_name, "featureMapShapes")==0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        int size = fields[i].length;
                        fMapShapes.reserve(size);
                        const int *fMap = static_cast<const int*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            fMapShapes.push_back(*fMap);
                            fMap++;
                        }
                    }
                    else if(strcmp(attr_name, "layerVariances")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        layerVariances.reserve(size);
                        const double *lVar = static_cast<const double*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            layerVariances.push_back(*lVar);
                            lVar++;
                        }
                    }
                }
                // Num layers should match the number of feature maps from which boxes are predicted.
                assert(numLayers > 0);
                assert((int)fMapShapes.size() == numLayers);
                assert(aspectRatios.size() > 0);
                assert(layerVariances.size() == 4);

                // Reducing the number of boxes predicted by the first layer.
                // This is in accordance with the standard implementation.
                vector<float> firstLayerAspectRatios;

                int numFirstLayerARs = 3;
                for(int i = 0; i < numFirstLayerARs; ++i){
                    firstLayerAspectRatios.push_back(aspectRatios[i]);
                }
                // A comprehensive list of box parameters that are required by anchor generator
                GridAnchorParameters boxParams[numLayers];
                for(int i = 0; i < numLayers ; i++)
                {
                    if(i == 0)
                        boxParams[i] = {minScale, maxScale, firstLayerAspectRatios.data(), (int)firstLayerAspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                    else
                        boxParams[i] = {minScale, maxScale, aspectRatios.data(), (int)aspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                }

                mPluginPriorBox = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(boxParams, numLayers), nvPluginDeleter);
                return mPluginPriorBox.get();
            }
            else if(concatIDs.find(std::string(layerName)) != concatIDs.end())
            {
                const int i = concatIDs[layerName];
                assert(mPluginFlattenConcat[i] == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(concatAxis[i], ignoreBatch[i]));
                return mPluginFlattenConcat[i].get();
            }
            else if(!strcmp(layerName, "_concat_priorbox"))
            {
                assert(mPluginConcat == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginConcat = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createConcatPlugin(2, true), nvPluginDeleter);
                return mPluginConcat.get();
            }
            else if(!strcmp(layerName, "_NMS"))
            {

                assert(mPluginDetectionOutput == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                 // Fill the custom attribute values to the built-in plugin according to the types
                for(int i = 0; i < nbFields; ++i)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "iouThreshold") == 0)
                    {
                        detectionOutputParam.nmsThreshold =(float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "numClasses") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.numClasses = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxDetectionsPerClass") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.topK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreConverter") == 0)
                    {
                        std::string scoreConverter(static_cast<const char*>(fields[i].data), fields[i].length);
                        if(scoreConverter=="SIGMOID")
                            detectionOutputParam.confSigmoid = true;
                    }
                    else if(strcmp(attr_name, "maxTotalDetections") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.keepTopK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreThreshold") == 0)
                    {
                        detectionOutputParam.confidenceThreshold = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                }
                mPluginDetectionOutput = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDDetectionOutputPlugin(detectionOutputParam), nvPluginDeleter);
                return mPluginDetectionOutput.get();
            }
            else
            {
              assert(0);
              return nullptr;
            }
        }

    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));

        if (!strcmp(layerName, "_PriorBox"))
        {
            assert(mPluginPriorBox == nullptr);
            mPluginPriorBox = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginPriorBox.get();
        }
        else if (concatIDs.find(std::string(layerName)) != concatIDs.end())
        {
            const int i = concatIDs[layerName];
            assert(mPluginFlattenConcat[i] == nullptr);
            mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(serialData, serialLength));
            return mPluginFlattenConcat[i].get();
        }
        else if (!strcmp(layerName, "_concat_priorbox"))
        {
            assert(mPluginConcat == nullptr);
            mPluginConcat = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginConcat.get();
        }
        else if (!strcmp(layerName, "_NMS"))
        {
            assert(mPluginDetectionOutput == nullptr);
            mPluginDetectionOutput = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginDetectionOutput.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char* name) override
    {
        return !strcmp(name, "_PriorBox")
            || concatIDs.find(std::string(name)) != concatIDs.end()
            || !strcmp(name, "_concat_priorbox")
            || !strcmp(name, "_NMS")
            || !strcmp(name, "mbox_conf_reshape");
    }

    // The application has to destroy the plugin when it knows it's safe to do so.
    void destroyPlugin()
    {
        for (unsigned i = 0; i < concatIDs.size(); ++i)
        {
            mPluginFlattenConcat[i].reset();
        }
        mPluginConcat.reset();
        mPluginPriorBox.reset();
        mPluginDetectionOutput.reset();
    }

    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginPriorBox{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginDetectionOutput{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginConcat{nullptr, nvPluginDeleter};
    std::unique_ptr<FlattenConcat> mPluginFlattenConcat[2]{nullptr, nullptr};
};

#endif
