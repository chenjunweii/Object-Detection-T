#ifndef FLT_TRT_UTILS
#define FLT_TRT_UTILS

#include <omp.h>
#include <mutex>
#include <unistd.h>
#include "base.h"
#include "box.hh"
#include <boost/format.hpp>      



inline int MatToTRTArray(Mat & m, vector <float> & data, int index = 0){

	int channel = m.channels();

	int height = m.size().height; int width = m.size().width;

    int volImg = channel * height * width;

	//#pragma omp parallel for
	
	for (int c = 0; c < channel; ++c){

		//#pragma omp parallel for

		for (unsigned j = 0, volChl = height * width; j < volChl; ++j) {

			//data[index * volImg + c * volChl + j] = (2.0 / 255.0) * static_cast <float> (m.data[j * channel]) - 1.0;
			data[index * volImg + c * volChl + j] = (2.0 / 255.0) * static_cast <float> (m.data[j * channel - c + 2]) - 1.0;


		}
	}

}



int MatVectorToTRTArray(vector <Mat> & ms, vector <float> & data){

}


inline int clip(int x, int lower, int upper) {
  
	return max(lower, min(x, upper));
}

std::vector <std::pair <int64_t, nvinfer1::DataType>> calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize){

    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;

    for (int i = 0; i < nbBindings; ++i){

        Dims dims = engine.getBindingDimensions(i);
        
		nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samples_common::volume(dims) * batchSize;
        
		sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}


void populateClassLabels(vector <string> & classes){

    auto fileName = "ssd_coco_labels.txt";

    std::ifstream labelFile(fileName);
    
	string line;
    
	int id = 0;
	
	while (getline(labelFile, line))
		
		classes[id++] = line;

    return;
}


void GetClassFromTXT(vector <string> & classes, string filename){

    std::ifstream ifs(filename.c_str());

	string line;
    
	int id = 0;
	
	while (getline(ifs, line))
		
		classes.emplace_back(line);

	ifs.close();

}

template <class T>

inline void acquire(vector <T> & out, deque <T> & q, mutex & m, bool & get, int nbatch, unsigned int wait, string comment){

	get = false;

	//printf("[*] Enter %s\n", comment.c_str());
	
	while (! get){

		while (q.empty() || (q.size() < nbatch)){
			
			//printf("[*] %s Queue is not enough : %d\n", comment.c_str(), q.size());

			usleep(wait);
		}

		get = m.try_lock();
	}

	for (int b = 0; b != nbatch; ++b){

		out.emplace_back(move(q.front()));
	
		q.pop_front();

	}
	
	m.unlock();
	
	q.shrink_to_fit();

	get = false;
	
	usleep(wait);

}

template <class T>

inline void emplace(vector <T> & in, deque <T> & q, mutex & m, bool & get, int nbatch, unsigned int wait){
	
	get = false;
	
	while (! get){

		get = m.try_lock();
		
		if (not get)
			
			usleep(wait);
	}

	for (int b = 0; b != nbatch; ++b)

		q.emplace_back(move(in[b]));
	
	m.unlock();

	in.clear();

	in.shrink_to_fit();
	
	get = false;

	usleep(wait);

	
}


inline void visualize(vector <Mat> & in, vector <flt::bboxes> & boxes, vector <string> & classes, float threshold){
	for (auto & box : boxes[0]){
		if (box.s >= threshold){
			Point ul(box.x, box.y);
			Point br(box.x1, box.y1);
			Point tp(box.x1, box.y1 - 5);
			rectangle(in[0], ul, br, cv::Scalar(0, 255, 0), 2);
			string text = boost::str(boost::format("%s : %f") % classes[box.c] % box.s);
			putText(in[0], text, tp, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,  cv::Scalar(0, 0, 255, 255), 1.5);
			cout << "[*] Visualize Detect Classes : " << classes[box.c] << endl;
		}
	}

	cv::imshow("Visualize", in[0]);

	cv::waitKey(int(33));

		//cv::waitKey(0);
}




#endif
