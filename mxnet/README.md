mxnet is compiled with protobuf 2.5 (8.0), sees ${mxnet}/deps, so if we want to compile a program with mxnet, the protobuf 2.5 in necessary






NDArray 用法

最好不要

NDArray nd;


nd = NDArray(Shape(3, 224, 224), ctx);

可以的話盡量

當下就創建好

NDArray nd(Shape(3, 224, 224), ctx);

或是

map <string, NDArray> nds;

nds["nd"] = NDArray(Shape(3, 224, 224), ctx);


