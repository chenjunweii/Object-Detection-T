
#include "src/base.h"
#include "src/utils.hh"
#include "src/loader.hh"
#include "src/detector.hh"

#include <thread>


//using namespace std;
//

int main(){

	int N = 1;

	const float visualizeThreshold = 0.5;

	static constexpr int OUTPUT_CLS_SIZE = 91;

	static constexpr int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

	char * serialized = nullptr;
	
	const char* OUTPUT_BLOB_NAME0 = "NMS";

	const char* INPUT_BLOB_NAME = "Input";
    
	string classes = "ssd_coco_labels.txt";
	
	IHostMemory * model;
	
	IRuntime * runtime;
	
	ICudaEngine * engine;
	
	PluginFactory plugin;
	
	IExecutionContext * context;
    
	vector <float> detectionOut(N * detectionOutputParam.keepTopK * 7);
    
	vector <int> keepCount(N);

	int size = 300;

    vector <float> data (N * 3 * size * size);

	//flt::load_serialized_model("mobilenet_v2_custom.trt", & serialized, & runtime, & engine, & context, & plugin);

	//Mat mm = imread("dog.jpg");

	//Mat m(Size(300, 300), CV_8UC3);

	ssd s("mobilenet_v2_lite.trt", classes, N, Size (size, size));
	
	string video = "TimeSquare.mp4";
	
	DetectType dt = DetectType::video;

	thread capture (& ssd::capture_thread, & s, ref(dt), ref(video));

    thread detect (& ssd::detect_thread, & s);

	s.post_thread(true);
    
	capture.join();

    detect.join();


}
