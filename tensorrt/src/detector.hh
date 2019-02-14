#ifndef FLT_TRT_DETECTOR_HH
#define FLT_TRT_DETECTOR_HH

#include <mutex>
#include <queue>
#include <memory>
#include <assert.h>
#include <unistd.h>
#include <condition_variable>
#include <boost/format.hpp>

#include "box.hh"
#include "base.h"
#include "struct.h"

class ssd{

public:

	deque <Mat> OriginalQueue, ReSizedQueue;
	
	deque <flt::bboxes> BoxesQueue;

	mutex mOriginalQueue, mResizedQueue, mBoxesQueue;

	//condition_variable cOriginalQueue, cResizedQueue, cBoxesQueue;

	DetectType detect_type;
	
	Size size, osize;

	map <string, bool> alive;
	
	unsigned int wait = 100;

	int nbatch, nclass;

	vector <string> classes;

	int bytesize;

	int inputIndex, outputIndex0, outputIndex1;

	const float visualizeThreshold = 0.5;

	//int OUTPUT_BBOX_SIZE;// = nclass * 4;

	char * serialized = nullptr;
	
	const char * OUTPUT_BLOB_NAME0 = "NMS";

	const char * INPUT_BLOB_NAME = "Input";
	
	cudaStream_t stream;
	
	IHostMemory * model;
	
	IRuntime * runtime;
	
	ICudaEngine * engine;
	
	PluginFactory plugin;
	
	IExecutionContext * context;
    
	vector <float> detectionOut;//(N * detectionOutputParam.keepTopK * 7);
    
	vector <int> keepCount;//(N);

    vector <float> data;// (N * 3 * 300 * 300);
		
	int nbBindings;// = engine->getNbBindings();

	vector <void *> buffers;//(nbBindings);
	
	vector <pair <int64_t, nvinfer1::DataType>> buffersSizes;// = calculateBindingBufferSizes(*engine, nbBindings, N);
	
	ssd(){};

	ssd(string filename, string _classes, int _nbatch, Size _size) : nbatch(_nbatch), size(_size){

		flt::load_serialized_model(filename, & serialized, & runtime, & engine, & context, & plugin);

		data = vector <float> (nbatch * 3 * size.width * size.height);
	
		detectionOut = vector <float> (nbatch * detectionOutputParam.keepTopK * 7);

		keepCount = vector <int> (nbatch);
		
		nbBindings = engine->getNbBindings();

		buffers = vector <void *> (nbBindings);

		buffersSizes = calculateBindingBufferSizes(*engine, nbBindings, nbatch);

		GetClassFromTXT(classes, _classes);

		for (int i = 0; i < nbBindings; ++i){

			auto bufferSizesOutput = buffersSizes[i];
			
			buffers[i] = samples_common::safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
		}

		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings().
		inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME),
			outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0),
			outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

		bytesize = nbatch * 3 * size.width * size.height * sizeof(float);

		CHECK(cudaStreamCreate(&stream));

	}


	~ssd(){

		cudaStreamDestroy(stream);
		CHECK(cudaFree(buffers[inputIndex]));
		CHECK(cudaFree(buffers[outputIndex0]));
		CHECK(cudaFree(buffers[outputIndex1]));
		context->destroy();
		engine->destroy();
		runtime->destroy();
		plugin.destroyPlugin();
		delete serialized;
	}

	inline int load(vector <Mat> & in){

		for (int i = 0; i != in.size(); ++i){

			MatToTRTArray(in[i], data, i);

		} 

		CHECK(cudaMemcpyAsync(buffers[inputIndex], & data[0], bytesize, cudaMemcpyHostToDevice, stream));

	}


	inline int unload(){

		CHECK(cudaMemcpyAsync(&detectionOut[0], buffers[outputIndex0], nbatch * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		
		CHECK(cudaMemcpyAsync(&keepCount[0], buffers[outputIndex1], nbatch * sizeof(int), cudaMemcpyDeviceToHost, stream));
		
		cudaStreamSynchronize(stream);

	}

	int capture_thread(DetectType & dt, string & filename){

		Mat frame, resized;
		
		VideoCapture capture;
		
		if (dt == DetectType::video){
			
			capture = VideoCapture(filename);

			cout << "is video" << endl;

		}
		
		if (!capture.isOpened()){

			cout << "not open" << endl;
			return -1;
		}

		detect_type = dt;

		osize = Size(int(capture.get(CV_CAP_PROP_FRAME_WIDTH)), int(capture.get(CV_CAP_PROP_FRAME_HEIGHT)));

		bool get = false;
		
		alive["capture"] = true;

		int idx = 0;
		
		while(true){
			
			guard();
		
			vector <Mat> frames, resizeds;

			cout << "[*] Capture Idx : " << idx << endl;

			idx += 1;

			while(OriginalQueue.size() > 20){
				
				usleep(wait * 5);
			
				if (not alive["capture"]) break;

			}

			capture >> frame;

			if(frame.empty()) break;

			resize(frame, resized, Size(size.width, size.height));
			
			//cvtColor(resized, resized, CV_BGR2RGB);

			resizeds.emplace_back(move(resized));

			frames.emplace_back(move(frame));
			
			emplace <Mat> (frames, OriginalQueue, mOriginalQueue, get, nbatch, wait);
			
			emplace <Mat> (resizeds, ReSizedQueue, mResizedQueue, get, nbatch, wait);
			
			if (not alive["capture"]) break;

		}
		
		alive["capture"] = false;
		
		return 0;	

	}

	int detect_thread(){

		bool get = false;

		vector <Mat> ins;

		int counter = 0;
		
		auto t_start = std::chrono::high_resolution_clock::now();
		
		alive["detect"] = true;

		while (true){
			
			guard();

			acquire <Mat> (ins, ReSizedQueue, mResizedQueue, get, nbatch, wait, "detect-Mat");
		
			auto t_start_inf = std::chrono::high_resolution_clock::now();

			load(ins);
			
			context->execute(nbatch, &buffers[0]);

			vector <flt::bboxes> boxes (nbatch);

			unload();

			convert(0, boxes[0]);
				auto t_end_inf = std::chrono::high_resolution_clock::now();
				float total_inf = std::chrono::duration <float, std::milli> (t_end_inf - t_start_inf).count();
				
				cout << "Time Elapse Inference: " << total_inf << endl;
			
			emplace <flt::bboxes> (boxes, BoxesQueue, mBoxesQueue, get, nbatch, wait);

			ins.clear();

			ins.shrink_to_fit();

			counter += 1;

			/*if (counter == 100){

				auto t_end = std::chrono::high_resolution_clock::now();
				
				float total = std::chrono::duration <float, std::milli> (t_end - t_start).count();
				
				cout << "Time Elapse : " << total << endl;

				break;
			}*/
			
			if (not alive["detect"])
				
				break;
		
		}
		
		alive["detect"] = false;

	};

inline int guard(){
	for (auto & i : alive){
		if (i.second == false){
			//printf("[!] Guard : %s is closed\n", i.first.c_str());
			for (auto & j : alive){
				j.second = false;
			}
		}
	}
}
void post_thread(bool viz){

	bool get = false;

	bool close = false;
	
	alive["post"] = true;
	
	while (true){
		
		guard();
		
		vector <Mat> frames;
		
		vector <flt::bboxes> boxes;
		
		cout << "[*] In Post Thread" << endl;

		acquire <flt::bboxes> (boxes, BoxesQueue, mBoxesQueue, get, 1, wait, "Post-Boxes");
		
		acquire <Mat> (frames, OriginalQueue, mOriginalQueue, get, 1, wait, "Post-Original");

		visualize(frames, boxes, classes, visualizeThreshold);

		//usleep(wait);
		//
		//
		/*

		frames.clear();

		boxes.clear();

		frames.shrink_to_fit();

		boxes.shrink_to_fit();

		*/
		
		if (not alive["post"])
			
			break;

	}

	alive["post"] = false;
}



inline void convert(int batch_id, flt::bboxes & boxes){

	#pragma omp parallel for

	for (int i = 0; i < keepCount[batch_id]; ++i){

		float * det = & detectionOut[0] + (batch_id * detectionOutputParam.keepTopK + i) * 7;

		if (det[2] < visualizeThreshold) continue;
		
		assert((int) det[1] < OUTPUT_CLS_SIZE);
		
		boxes.emplace_back(move(flt::bbox(det, osize)));
	}
}




};

#endif
