#include <string>
#include "detector.h"

using namespace std;

void detector::detect_image(string & str){

	cv::Mat image = cv::imread(str, CV_LOAD_IMAGE_COLOR);
	
	int height = image.size().height;

	int width = image.size().width;
	
	cv::Mat resized;
	
	vector <cv::Mat> inputs;

	Size size_ (width, height);

	cv::resize(image, resized, cv::Size(size.w, size.h));
	
	cv::cvtColor(resized, resized, CV_BGR2RGB);

	inputs.emplace_back(resized);

	MatVector_to_NDArray(args[0]["data"], inputs, *ctx);

	args[0]["data"] -= nds["mean"];

	NDArray::WaitAll();

	Executor * EX = net.SimpleBind(*ctx, args[0], grad, req, auxs[0]);

	EX->Forward(false);
	
	NDArray::WaitAll();

	vector <vector <bbox>> boxes;

	cout << EX->outputs.size() << endl;

	cout << EX->outputs[0] << endl;
	
	convert(EX->outputs, boxes, size_);
	
	NDArray::WaitAll();

	visualize(image, boxes);

}
