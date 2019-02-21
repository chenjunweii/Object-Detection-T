#ifndef FLT_HDF5_HH
#define FLT_HDF5_HH

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <H5Cpp.h>  
#include "fhdf5.h"
#include <hdf5_hl.h>

using namespace std;
using namespace H5;

/*
 *	H5F_ACC_TRUNC : Truncate file, if it already exists, erasing all data previously stored in the file.
 *	H5F_ACC_EXCL : Fail if file already exists. 	
 */

flt::fhdf5::fhdf5(const char * f) : filename(f){
	
};

flt::fhdf5::fhdf5(string f) : filename(f.c_str()){
	
};
flt::fhdf5::fhdf5(const char * f, Context *context) : filename(f){

	ctx = context;
	
};

inline void flt::fhdf5::read(){

	file = H5File(filename, H5F_ACC_RDONLY);
}

inline flt::fhdf5::fhdf5(){};
inline void flt::fhdf5::open(){

	file = H5File(filename, H5F_ACC_TRUNC);
}

inline void flt::fhdf5::create(){
	
	hid_t id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

}

inline void flt::fhdf5::close(){
	
	file.close();
}

inline void flt::fhdf5::open_exist(){

	file = H5File(filename, H5F_ACC_RDWR);
}

inline void flt::fhdf5::list_object(){
	
	herr_t status = H5Ovisit(file.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, iterate_all_object, nullptr);
}

inline void flt::fhdf5::load_keras_all(){
	
	herr_t status = H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, iterate_keras_group, (void *) &keras);
}

inline void flt::fhdf5::load_keras(string node){
	
	vector <void *> vvoid = {(void *) &nddata, (void *) ctx};
	
	herr_t status = H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, iterate_keras_group, (void *) &keras);
}
inline void flt::fhdf5::load_weight(){
	
	vector <void *> vvoid = {(void *) &nddata, (void *) ctx};

	herr_t status = H5Ovisit(file.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, iterate_weight_object, (void *) &vvoid);
		
}

flt::fhdf5::~fhdf5 (){
	
	file.close();
};
inline void flt::fhdf5::save(map <string, vector <float>> &m){

}


inline void flt::fhdf5::save_NDArray(map <string, NDArray> &ndarg){
	
	for(auto &n : ndarg){
		
		vector <mx_uint> shape = n.second.GetShape();
		
		int rank = shape.size();

		int size = Shape(shape).Size();

		hsize_t dimensions[rank];

		for (int i = 0; i != rank; ++i)

			dimensions[i] = shape[i];
		
		DataSpace fspace(rank, dimensions);
		
		DataSet dataset = file.createDataSet(n.first, PredType::NATIVE_FLOAT, fspace);
			
		vector <float> fdata(size);

		n.second.SyncCopyToCPU(fdata.data(), size);
		
		NDArray::WaitAll();
	
		dataset.write(fdata.data(), PredType::NATIVE_FLOAT);
	}

}


herr_t flt::fhdf5::iterate_all_object (hid_t location_id, const char * name, const H5O_info_t * info, void * p){

    printf ("/");               /* Print root group in object path */

    if (name[0] == '.')         /* Root group, do not print '.' */

        printf ("  (Group)\n");

    else if (info->type == H5O_TYPE_GROUP){

        printf ("%s  (Group)\n", name);

    }

    else if (info->type == H5O_TYPE_DATASET){

        
		hid_t weight_id = H5Dopen1(location_id, name);
		
		hid_t space = H5Dget_space(weight_id);
		
		DataSpace dataspace(space);
		
		int rank = dataspace.getSimpleExtentNdims();
		
		hsize_t dimensions[rank];
      	
		int ndims = dataspace.getSimpleExtentDims(dimensions, NULL);
		
		int size = 1;
		
		vector <mx_uint> shape(rank, 0);
		
		for (int i = 0; i != rank; ++i){
		
			size *= dimensions[i];
			
			shape[i] = dimensions[i];
		}

		cout << name << "   (Dataset) : " << Shape(shape) << endl;

        printf ("%s  (Dataset)\n", name);
    }

    else if (info->type == H5O_TYPE_NAMED_DATATYPE){

        printf ("%s  (Datatype)\n", name);

    }

    else

        printf ("%s  (Unknown)\n", name);


    return 0;
}

herr_t flt::fhdf5::iterate_keras_group (hid_t locationid, const char * name, const H5L_info_t * info, void * p){

	map <string, map <string, vector <float>>> *pointer = (map <string, map <string, vector <float>>> *) p;

    hsize_t nobj;

    hid_t groupid;

    if (info->type == H5O_TYPE_GROUP){

		groupid = H5Gopen1(locationid, name); // open location as group

        H5Gget_num_objs(groupid, &nobj); // check if it contains dataset

		if (nobj > 0){
			
			hid_t weight_id = H5Dopen1(groupid, "param_0");

			hid_t bias_id = H5Dopen1(groupid, "param_1");
			
			//hsize_t weight_size = H5Dget_storage_size(weight_id);
			
			//hsize_t bias_size = H5Dget_storage_size(bias_id);
			
			hid_t weight_space_id = H5Dget_space(weight_id);

			hid_t bias_space_id = H5Dget_space(bias_id);

			DataSpace weight_space(weight_space_id);

			DataSpace bias_space(bias_space_id);


			int weight_rank = weight_space.getSimpleExtentNdims();
			
			int bias_rank = bias_space.getSimpleExtentNdims();

			hsize_t weight_dimensions[weight_rank];

			hsize_t bias_dimensions[bias_rank];
      	
			int weight_dimensions_i = weight_space.getSimpleExtentDims(weight_dimensions, NULL);

			int bias_dimensions_i = bias_space.getSimpleExtentDims(bias_dimensions, NULL);

			int weight_size = 1;
			
			int bias_size = 1;

			for(int i = 0; i != weight_rank; ++i)
				
				weight_size *= weight_dimensions[i];

			for(int i = 0; i != bias_rank; ++i)

				bias_size *= bias_dimensions[i];

			(*pointer)["weight"][name] = vector <float> (weight_size);

			(*pointer)["bias"][name] = vector <float> (bias_size);
			
			//cout << "Layer : " << name << endl;

			//cout << "b size : " << bias_size << endl;
			//cout << "w size : " << weight_size << endl;
			H5LTread_dataset(groupid, "param_0", H5T_NATIVE_FLOAT, (*pointer)["weight"][name].data());
			
			H5LTread_dataset(groupid, "param_1", H5T_NATIVE_FLOAT, (*pointer)["bias"][name].data());

		}

    }

    return 0;
}

herr_t flt::fhdf5::iterate_weight_object (hid_t location_id, const char * name, const H5O_info_t * info, void * vdata){
	
	vector <void *> * vvoid = (vector <void *> *) vdata; 
	
	map <string, NDArray> * nd = (map <string, NDArray> *) (*vvoid)[0]; // for our own purpose
	
	Context *vctx  = (Context *) (*vvoid)[1];
	
    hsize_t nobj;

    hid_t groupid;

    if (info->type == H5O_TYPE_DATASET){
		
		hid_t weight_id = H5Dopen1(location_id, name);
		
		hid_t space = H5Dget_space(weight_id);
		
		DataSpace dataspace(space);
		
		int rank = dataspace.getSimpleExtentNdims();
		
		hsize_t dimensions[rank];
      	
		int ndims = dataspace.getSimpleExtentDims(dimensions, NULL);
		
		int size = 1;
		
		vector <mx_uint> shape(rank, 0);
		
		for (int i = 0; i != rank; ++i){
		
			size *= dimensions[i];
			
			shape[i] = dimensions[i];
		}
		
		(*nd)[name] = NDArray(Shape(shape), (*vctx));
		
		vector <float > v(size);
		
		H5LTread_dataset(location_id, name, H5T_NATIVE_FLOAT, v.data());
			
		(*nd)[name].SyncCopyFromCPU(v.data(), size);

		herr_t hclose = H5Dclose(weight_id);

    }

    return 0;
}
#endif
