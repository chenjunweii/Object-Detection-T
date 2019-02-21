#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

int main(){
	
	auto receive = VideoCapture("test.mp4");

	auto send = VideoWriter("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=0.0.0.0 port    =5000", CAP_GSTREAMER, 0, 20, Size(320, 240), true);
	

	if(!receive.isOpened())  // check if we succeeded
		return -1;

	Mat frame;

	while (true) {

		receive >> frame; // get a new frame from camera

		resize(frame, frame, Size(320, 240));

		send.write(frame);

	}

}
