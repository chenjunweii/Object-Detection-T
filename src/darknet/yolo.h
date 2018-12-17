#ifndef FLT_DARKNET_YOLO_H
#define FLT_DARKNET_YOLO_H

#include <string>
#include <vector>
#include "image.hh"
#include <opencv2/opencv.hpp>

using namespace std;

const int netW = 416;
const int netH = 416;
const int classes = 80;
const float thresh = 0.5;
const float hier_thresh = 0.5;
const float nms = 0.45;
const int numBBoxes = 3;
const int numAnchors = 9;
const int relative = 1;

typedef struct { float x,y,w,h; } box;

struct detection {
    box bbox;
    int classes;
    float * prob;
    float * mask;
    float objectness;
    int sort_class;
};

struct layer {
    int batch;
    int total;
    int n, c, h ,w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs, outputs;
    float * mask;
    float * biases;
    float * output;
    float * output_gpu;

	layer(){};

	layer(string l, map <string, vector <float>> & fs, map <string, vector <int64_t>> & out_shape){

		output = fs[l + "_output"].data();
		mask = fs[l + "_mask"].data();
		biases = fs[l + "_bias"].data();

		out_h = out_shape[l + "_output"][2];
		out_w = out_shape[l + "_output"][3];

	}
};

layer make_yolo_layer(int batch,int w,int h,int n,int total,int classes);

void free_yolo_layer(layer l);

void forward_yolo_layer_gpu(const float* input,layer l, float* output);

//detection * get_detections(vector<Blob<float>*> blobs,int img_w,int img_h,int* nboxes);

detection * get_detections(vector <vector <float>> layer_out, vector <int> lw, vector <int> lh, int w, int h, int * nboxes);

void free_detections(detection *dets,int nboxes);

#endif
