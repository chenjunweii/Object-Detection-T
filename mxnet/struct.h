#ifndef STRUCT_H
#define STRUCT_H

enum DetectType {
	camera,
	image,
	video
};

enum Type {
	real,
	norm
};

struct Size {

	int w = 0;

	int h = 0;

	Size(int _w, int _h) : w(_w), h(_h) {};

	Size() {};

};

#endif
