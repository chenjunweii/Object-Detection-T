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

typedef struct {
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
} detection;

typedef struct {
    int batch;
    int total;
    int n,c,h,w;
    int out_n,out_c,out_h,out_w;
    int classes;
    int inputs,outputs;
    int *mask;
    float* biases;
    float* output;
    float* output_gpu;
} layer;

layer make_yolo_layer(int batch,int w,int h,int n,int total,int classes);

void free_yolo_layer(layer l);

//detection * get_detections(vector<Blob<float>*> blobs,int img_w,int img_h,int* nboxes);

void free_detections(detection *dets,int nboxes);

#endif
