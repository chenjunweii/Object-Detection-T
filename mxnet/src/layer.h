#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <string>
#include <mxnet-cpp/MxNetCpp.h>

using namespace std;
using namespace mxnet::cpp;

namespace flt{
	
	namespace mx{

		namespace layer{

			inline void conv(char * prefix,
				char * name, 
				char * in,
				map <string, Symbol> *neurons,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				int filter,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				bool isRelu = true,
				bool noBias = false);

			inline Symbol conv(string snode,
				Symbol &input,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				int filter,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				bool isRelu = true,
				bool noBias = false);
			
			/* no bias , integrated with weight*/

			inline void conv(char * name, 
				char * input,
				map <string, Symbol> *neurons,
				map <string, Symbol> *weight,
				int filter,
				Shape kerneli = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				bool isRelu = true,
				bool noBias = false);
			
			inline Symbol conv(string snode,
				Symbol &input,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				map <string, Symbol> *aux,
				int filter,
				string act,
				bool isTraining,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				bool isRelu = true,
				bool noBias = false);

			
			inline void deconv(char * prefix,
				char * name,
				char * in,
				map <string, Symbol> *neurons,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				int filter,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				Shape adj = Shape(),
				Shape dilate = Shape(),
				bool isRelu = true,
				bool noBias = true);

			inline Symbol deconv(string snode,
				Symbol &input,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				map <string, Symbol> *aux,
				int filter,
				string act,
				bool isTraining,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				Shape adj = Shape(),
				Shape dilate = Shape(),
				bool isRelu = true,
				bool noBias = true);
			
			inline Symbol deconv(string snode,
				Symbol &input,
				map <string, Symbol> *weight,
				map <string, Symbol> *bias,
				int filter,
				Shape kernel = Shape(3,3),
				Shape stride = Shape(1,1),
				Shape pad = Shape(1,1),
				Shape adj = Shape(),
				Shape dilate = Shape(),
				bool isRelu = true,
				bool noBias = true);
			inline void fullyconnected(char * prefix,
					char * name,
					char * in,
					map <string, Symbol> *neurons,
					map <string, Symbol> *weight,
					map <string, Symbol> *bias,
					int nout);
			
			inline Symbol fullyconnected(string snode,
					Symbol &input,
					map <string, Symbol> *weight,
					map <string, Symbol> *bias,
					int nout);

			inline Symbol fullyconnected(string snode,
					Symbol &input,
					map <string, Symbol> *weight,
					map <string, Symbol> *bias,
					map <string, Symbol> *aux,
					string act,
					bool isTraining,
					int nout);
			inline void maxpool(char * prefix,
					char * name,
					char * in,
					map <string, Symbol> *neurons,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(1,1),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);

			inline void maxpool(char * name,
					char * in,
					map <string, Symbol> *neurons,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(1,1),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);
			
			inline Symbol maxpool(string snode,
					Symbol &in,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(1,1),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);
			
			inline void avgpool(char * name,
					char * in,
					map <string, Symbol> *neurons,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(0,0),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);
			
			inline void avgpool(char * prefix,
					char * name,
					char * in,
					map <string, Symbol> *neurons,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(0,0),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);

			inline Symbol avgpool(string snode,
					Symbol &in,
					Shape kernel = Shape(2,2),
					Shape stride = Shape(2,2),
					Shape pad = Shape(0,0),
					PoolingPoolingConvention method = PoolingPoolingConvention::kValid);

			inline void concat(char * prefix,
					char * name,
					vector <Symbol> *v,
					map <string, Symbol> *m,
					int dimension);
			
			inline Symbol concat(string snode,
					vector <Symbol> *v,
					int dimension);

			inline Symbol concat(string snode,
					vector <Symbol> &&v,
					int dimension);
			inline Symbol concat(char * prefix,
					char * name,
					vector <Symbol> *v,
					int dimension);
		
		} /* layer */

	} /* mx */

} /* flt */

#endif
