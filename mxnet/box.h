#ifndef BOX_H
#define BOX_H

struct bbox {

	public:

		bbox();
		
		bbox(vector <float> & fbbox, cv::Size & size);
		
		bbox(float c, float s, vector <float> & fbbox, cv::Size & dsize, cv::Size & osize);

		bbox(float c, float s, float x, float y, float x1, float y1);

		int c = 0;

		float s = 0;

		int x = 0;

		int y = 0;

		int x1 = 0;

		int y1 = 0;

};

#endif
