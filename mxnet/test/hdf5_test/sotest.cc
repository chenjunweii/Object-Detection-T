#include <iostream>
#include <H5Cpp.h>


using namespace std;
using namespace H5;
int main(){
	string filename = "test_create.h5";

  herr_t status;

  hid_t file = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);


	cout << "xxx " << endl;
}
