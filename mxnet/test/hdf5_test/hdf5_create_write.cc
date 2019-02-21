#include <iostream>
#include <string>
#include <H5Cpp.h>

using namespace H5;
using namespace std;

int main(){

  const int NX = 5;
  const int NY = 6;
  const int RANK = 2;

  int i, j;

  int data[NX][NY];          // buffer for data to write

  for (j = 0; j < NX; j++){

    for (i = 0; i < NY; i++)

        data[j][i] = i + j;

  }

  /*
  * 0 1 2 3 4 5
  * 1 2 3 4 5 6
  * 2 3 4 5 6 7
  * 3 4 5 6 7 8
  * 4 5 6 7 8 9
  */

  string filename = "test_create.h5";

  string datasetname = "intarray";

  H5File file(filename, H5F_ACC_TRUNC);

  hsize_t dimsf[2];

  dimsf[0] = NX; dimsf[1] = NY;

  DataSpace dataspace(RANK, dimsf);

  IntType datatype(PredType::NATIVE_INT);

  datatype.setOrder(H5T_ORDER_LE);

  DataSet dataset = file.createDataSet(datasetname, datatype, dataspace);

  dataset.write(data, PredType::NATIVE_INT);
  
}
