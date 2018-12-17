#ifndef UTILS_HH
#define UTILS_HH

#include "tvm.h"

inline int clip(int x, int lower, int upper) {
  
	return max(lower, min(x, upper));
}

#endif
