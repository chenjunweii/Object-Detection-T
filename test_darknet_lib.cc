#include <iostream>
#include "src/darknet/image.hh"


#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace flt::darknet;


void show10(float * f, string comment){

	cout << comment << " : [";

	for (int i = 0; i != 10; ++i)

		cout << f[i] << ", ";

	cout << "]" << endl;
}

void MatToFloatArray(Mat & mat, vector <float> & farray){

	int h = mat.size().height;

	int w = mat.size().width;

	int channel = mat.channels();

	int cbase, hbase = 0;

	for (int c = 0; c < channel; ++c) {

		cbase = c * h * w;

		for (int i = 0; i < h; ++i) {

			hbase = h * i;

			for (int j = 0; j < w; ++j) {

					farray[cbase + hbase + j] = (static_cast <float> (mat.data[(hbase + j) * channel + c])) / 255.0;
			}
		}
		
	}
}


void _resize(Mat & in, Mat & out, int w, int h){

	/*

    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    
	int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);


    return resized;

}
	*/

}

int main(){
	
	Mat mat = imread("dog.jpg");

	Mat resized;

	Mat resized2(Size(416, 416), CV_32FC3);

	resize(mat, resized, Size(416, 416));

	_resize(mat, resized, 416, 416);

	Size size = mat.size();

	vector <float> fdata (416 * 416 * 3);

	MatToFloatArray(resized, fdata);

	int w = size.width; int h = size.height;
	
	//IplImage * pointer = (IplImage *) & mat;
	//
	IplImage ipl = IplImage(mat);

	show10((float*)ipl.imageData, "IpLImage");

	image yim = load_image_color("dog.jpg", 0, 0);

	image ycustom = load_image(mat, 0, 0, 3);

	show10(yim.data, "yolo load_image");

	show10((float*)resized.data, "opencv load_image");
	
	show10(fdata.data(), "opencv load_image to float");
	
	show10(ycustom.data, "Mat To Image");
    
	//image yim_sized = letterbox_cv2(im, 416, 416);



}
