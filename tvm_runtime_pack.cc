/*!
 * \brief This is an all in one tvm_root runtime file.
 *
 *   You only have to use this file to compile libtvm_root_runtime to
 *   include in your project.
 *
 *  - Copy this file into your project which depends on tvm_root runtime.
 *  - Compile with -std=c++11
 *  - Add the following include path
 *     - /path/to/tvm_root/include/
 *     - /path/to/tvm_root/3rdparty/dmlc-core/include/
 *     - /path/to/tvm_root/3rdparty/dlpack/include/
 *   - Add -lpthread -ldl to the linked library.
 *   - You are good to go.
 *   - See the Makefile in the same folder for example.
 *
 *  The include files here are presented with relative path
 *  You need to remember to change it to point to the right file.
 *
 */
#include "tvm_root/src/runtime/c_runtime_api.cc"
#include "tvm_root/src/runtime/cpu_device_api.cc"
#include "tvm_root/src/runtime/workspace_pool.cc"
#include "tvm_root/src/runtime/module_util.cc"
#include "tvm_root/src/runtime/module.cc"
#include "tvm_root/src/runtime/registry.cc"
#include "tvm_root/src/runtime/file_util.cc"
#include "tvm_root/src/runtime/threading_backend.cc"
#include "tvm_root/src/runtime/thread_pool.cc"
#include "tvm_root/src/runtime/ndarray.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "tvm_root/src/runtime/dso_module.cc"
#include "tvm_root/src/runtime/system_lib_module.cc"

// Graph runtime
#include "tvm_root/src/runtime/graph/graph_runtime.cc"

// Uncomment the following lines to enable RPC
// #include "tvm_root/src/runtime/rpc/rpc_session.cc"
// #include "tvm_root/src/runtime/rpc/rpc_event_impl.cc"
// #include "tvm_root/src/runtime/rpc/rpc_server_env.cc"

// These macros enables the device API when uncommented.
#define tvm_root_CUDA_RUNTIME 1
#define tvm_root_METAL_RUNTIME 1
#define tvm_root_OPENCL_RUNTIME 1

// Uncomment the following lines to enable Metal
// #include "tvm_root/src/runtime/metal/metal_device_api.mm"
// #include "tvm_root/src/runtime/metal/metal_module.mm"

// Uncomment the following lines to enable CUDA
#include "tvm_root/src/runtime/cuda/cuda_device_api.cc"
#include "tvm_root/src/runtime/cuda/cuda_module.cc"

// Uncomment the following lines to enable OpenCL
// #include "tvm_root/src/runtime/opencl/opencl_device_api.cc"
// #include "tvm_root/src/runtime/opencl/opencl_module.cc"
