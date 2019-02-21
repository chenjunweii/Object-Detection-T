#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>

using namespace std;
using namespace mxnet::cpp;


int main(){
	
	Shape s(1,2,3);

	cout << s.total();
}
