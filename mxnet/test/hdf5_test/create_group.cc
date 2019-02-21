#include <iostream>
#include <string>
#include <H5Cpp.h>

using namespace H5;
using namespace std;

const int RANK = 2;

herr_t file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo,
    void *opdata);

int main(){

  hsize_t dims[2];

  hsize_t cdims[2];

  string filename = "test_create_group.h5";

  H5File * file = new H5File(filename.c_str(), H5F_ACC_TRUNC);

  Group * group = new Group(file->createGroup("/Data"));

  dims[0] = 1000;

  dims[1] = 20;

  cdims[0] = 20;

  cdims[1] = 20;

  DataSpace *dataspace = new DataSpace(RANK, dims);

  DSetCreatPropList ds_creatplist;

  ds_creatplist.setChunk(2, cdims);

  ds_creatplist.setDeflate(6);

  DataSet * dataset = new DataSet(file->createDataSet(
    "/Data/Compressed_Data", PredType::NATIVE_INT,
    *dataspace, ds_creatplist));

  delete dataset;
  delete dataspace;


  // Create the second dataset

  dims[0] = 500;
  dims[1] = 20;
  dataspace = new DataSpace(RANK, dims);
  dataset = new DataSet(file->createDataSet("/Data/Float_Data", PredType::NATIVE_FLOAT, *dataspace));

  delete dataset;
  delete dataspace;
  delete group;
  delete file;

  file = new H5File(filename, H5F_ACC_RDWR);

  group = new Group(file->openGroup("Data"));

  dataset = new DataSet(group->openDataSet("Compressed_Data"));

  cout << "dataset \"/Data/Compressed_Data\" is open" << endl;

  delete dataset;

  file->link(H5L_TYPE_HARD, "Data", "Data_new");

  dataset = new DataSet(file->openDataSet("/Data_new/Compressed_Data"));

  cout << "dataset is open" << endl;

  delete dataset;

  herr_t idx = H5Literate(file->getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);

  /*

  unlink name "Data" and use iterator to see the name;

  */


  cout << "Unlinking ... " << endl;

  file->unlink("Data");

  cout << "iterate again ... " << endl;

  idx = H5Literate(file->getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);

  delete group;
  delete file;

}

herr_t file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
{
    hid_t group;

    /*
     * Open the group using its name.
     */
    group = H5Gopen2(loc_id, name, H5P_DEFAULT);

    /*
     * Display group name.
     */
    cout << "Name : " << name << endl;

    H5Gclose(group);
    return 0;
}
