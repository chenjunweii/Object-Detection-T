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

	condition_variable cOriginalQueue, cResizedQueue, cBoxesQueue;

	DetectType detect_type;
	
	Size size, osize;

	map <string, bool> alive;
	
	unsigned int wait = 50;

	int nbatch, nclass;

	vector <string> classes;

	int bytesize;

	int inputIndex, outputIndex0, outputIndex1;

	const float visualizeThreshold = 0.5;

	int OUTPUT_BBOX_SIZE;// = nclass * 4;

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

		flt::load_serialized_model("mobilenet_v2_custom.trt", & serialized, & runtime, & engine, & context, & plugin);

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


	inline int load(Mat & in){
	
		CHECK(cudaMemcpyAsync(buffers[inputIndex], & data[0], bytesize, cudaMemcpyHostToDevice, stream));
	
	}
		
	inline int preprocess(Mat in, Mat out, int idx){

		resize(in, out, size);

		cv::cvtColor(out, out, CV_BGR2RGB);

		//MatToTRTArray(out, data, idx);

	}

	int detect_image(string filename){

		std::cout << " Data Size  " << data.size() << std::endl;
	}

	int capture_thread(DetectType & dt, string & filename){

		cout << "in Capture Thread " << endl;
		
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

			cout << "[*] Capture Idx : " << idx << endl;

			idx += 1;

			while(OriginalQueue.size() > 15){
				//cout << "[*]  OriginalQueue : " << OriginalQueue.size() << endl;
				usleep(wait * 10);

			}

			capture >> frame;

			if(frame.empty())
				break;

			resize(frame, resized, Size(size.width, size.height));
			
			cvtColor(resized, resized, CV_BGR2RGB);
			
			get = false;
			
			while (! get){
				get = mOriginalQueue.try_lock();
				if (not get)
					usleep(wait);
			}
			OriginalQueue.emplace_back(move(frame));
			mOriginalQueue.unlock();

			get = false;
			while (! get){
				get = mResizedQueue.try_lock();
				if (not get)
					usleep(wait);
			}

			ReSizedQueue.emplace_back(move(resized));
			mResizedQueue.unlock();

			usleep(wait);
			if (not alive["capture"])
				break;

			guard();
		}
		alive["capture"] = false;
		
		return 0;	

	}

	int detect_thread(){

		bool get = false;

		vector <Mat> ins;

		while (true){

			acquire <Mat> (ins, ReSizedQueue, mResizedQueue, get, nbatch, wait, "detect-Mat");

			load(ins);
			
			auto t_start = std::chrono::high_resolution_clock::now();
			
			context->execute(nbatch, &buffers[0]);
			
			auto t_end = std::chrono::high_resolution_clock::now();
			
			float total = std::chrono::duration <float, std::milli> (t_end - t_start).count();

			vector <flt::bboxes> boxes (nbatch);

			unload();

			convert(0, boxes[0]);

			cout << "Time Elapse : " << total << endl;
			
			emplace <flt::bboxes> (boxes, BoxesQueue, mBoxesQueue, get, nbatch, wait);

			ins.clear();

			ins.shrink_to_fit();

			guard();
		
		}

	};

inline int guard(){
	for (auto & i : alive){
		if (i.second == false){
			cout << "[*] Guard Close " << i.first << endl;
			for (auto & j : alive){
				j.second = false;
			}
		}
	}
}
void post_thread(bool viz){

	bool get = false;
	
	alive["capture"] = true;
	
	vector <Mat> frames;
		
	vector <flt::bboxes> boxes;
	
	while (true){

		cout << "[*] In Post Thread" << endl;

		acquire <flt::bboxes> (boxes, BoxesQueue, mBoxesQueue, get, 1, wait, "Post-Boxes");
		
		acquire <Mat> (frames, OriginalQueue, mOriginalQueue, get, 1, wait, "Post-Original");
		//
		//
		//
		/*
		
		get = false;
		
		while (! get){
			while (BoxesQueue.empty()){
				usleep(wait);
			}
			get = mBoxesQueue.try_lock();
		}

		//boxes = move(BoxesQueue.front());

		BoxesQueue.pop_front();
		BoxesQueue.shrink_to_fit();
		mBoxesQueue.unlock();
		get = false;

		while (! get){
			get = mOriginalQueue.try_lock();
			usleep(wait);
		}

		
		
		get = false;
		//frame = move(OriginalQueue.front());
		OriginalQueue.pop_front();
		OriginalQueue.shrink_to_fit();
		mOriginalQueue.unlock();
		//if (viz)
		//	visualize(frame, boxes, 0.5, 33);
		//
		*/

		visualize(frames, boxes, classes, visualizeThreshold);

		//usleep(wait);

		frames.clear();

		boxes.clear();

		frames.shrink_to_fit();

		boxes.shrink_to_fit();
		
		guard();
		
		if (not alive["capture"])
			
			break;

	}
}



inline void convert(int batch_id, flt::bboxes & boxes){

	for (int i = 0; i < keepCount[batch_id]; ++i){

		float * det = & detectionOut[0] + (batch_id * detectionOutputParam.keepTopK + i) * 7;

		if (det[2] < visualizeThreshold) continue;

		// Output format for each detection is stored in the below order
		// [image_id, label, confidence, xmin, ymin, xmax, ymax]
		assert((int) det[1] < OUTPUT_CLS_SIZE);
		
		boxes.emplace_back(move(flt::bbox(det, osize)));
		
		//string ppmid = "123";
		//
		//cout << "Classes : " << classes[int(det[1])] << endl;


		
		//printf("Detected %s in the image %d (%s) with confidence %f%% and coordinates (%f,%f),(%f,%f).\nResult stored in %s.\n", CLASSES[int(det[1])].c_str(), int(det[0]), ppmid, det[2] * 100.f, det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * , storeName.c_str());

	}
}




};

#endif
