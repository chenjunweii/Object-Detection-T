# Tuner

目前看起來 TVM 是可以直接從 mxnet 的 json 讀取 symbol，並從 params 讀取參數

使用  mx.symbol.load_checkpoint 這個函數讀取出來的會長這個樣子

```python
out, arg_params, aux_params = mx.model.load_checkpoint(network, epochs)
```

arg_params 和 aux_params 這兩個 python 字典的 key 不會有 "arg:" 和 "aux:" ，我看了一下範例讀取出來的雖然只有一個字典，但是裡面也沒有 "arg:" 和 "aux:"

# Test

目前加入了 test_symbol

## 有些 symbol 是不能 tune 的

https://docs.tvm.ai/nnvm_top.html



Python 的 TVM

可以把 MXnet 的 Model 用 compile 轉成 nnvm 版本的，詳細可以參考

https://docs.tvm.ai/tutorials/nnvm/deploy_ssd.html



## Dynamic Library vs. System Module

TVM provides two ways to use the compiled library. You can checkout [prepare_test_libs.py](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy/prepare_test_libs.py) on how to generate the library and [cpp_deploy.cc](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy/cpp_deploy.cc) on how to use them.

- Store library as a shared library and dynamically load the library into your project.
- Bundle the compiled library into your project in system module mode.

Dynamic loading is more flexible and can load new modules on the fly. System module is a more `static` approach. We can use system module in places where dynamic library loading is banned.

[Next ](https://docs.tvm.ai/deploy/android.html)[ Previous](https://docs.tvm.ai/deploy/index.html)

 

https://docs.tvm.ai/deploy/cpp_deploy.html

# How to Deploy

https://github.com/dmlc/tvm/tree/master/apps/howto_deploy

```python
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    # Compile library as dynamic library
    fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
    dylib_path = os.path.join(base_path, "test_addone_dll.so")
    fadd_dylib.export_library(dylib_path)

    # Compile library in system library mode
    fadd_syslib = tvm.build(s, [A, B], "llvm --system-lib", name="addonesys")
    syslib_path = os.path.join(base_path, "test_addone_sys.o")
    fadd_syslib.save(syslib_path)

```

```python
 def load_model(network):
 
     params = mx.nd.load(network + '.params') # must be deploy net
     network = mx.sym.load(network + '.json')                                             
     batch_shape = (1, 3, 224, 224)
     return network, params, batch_shape


net, params, input_shape = load_model('test') # params => mx.NDArray
net, params = nnvm.frontend.from_mxnet(net, params) # return params => tvm.NDArray                                                                                                                                  

```

```python
nnvm.compiler.save_param_dict(params)
Save parameter dictionary to binary bytes.

The result binary bytes can be loaded by the GraphModule with API “load_params”.

Parameters:	params (dict of str to NDArray) – The parameter dictionary.
Returns:	param_bytes – Serialized parameters.
Return type:	bytearray
Examples

# compile and save the modules to file.
graph, lib, params = nnvm.compiler.build(
   graph, target, shape={"data", data_shape}, params=params)
module = graph_runtime.create(graph, lib, tvm.gpu(0))
# save the parameters as byte array
param_bytes = nnvm.compiler.save_param_dict(params)
# We can serialize the param_bytes and load it back later.
# Pass in byte array to module to directly set parameters
module["load_params"](param_bytes)
nnvm.compiler.load_param_dict(param_bytes)
Load parameter dictionary to binary bytes.

Parameters:	param_bytes (bytearray) – Serialized parameters.
Returns:	params – The parameter dictionary.
Return type:	dict of str to NDArray
```



#Load Model

```python
tvm.module.load(path, fmt='')
This function will automatically call cc.create_shared if the path is in format .o or .tar

def load(path, fmt=""):
    """Load module from file.
    Parameters
    ----------
    path : str
        The path to the module file.
    fmt : str, optional
        The format of the file, if not specified
        it will be inferred from suffix of the file.
    Returns
    -------
    module : Module
        The loaded module
    Note
    ----
    This function will automatically call
    cc.create_shared if the path is in format .o or .tar
    """
    # High level handling for .o and .tar file.
    # We support this to be consistent with RPC module load.
    if path.endswith(".o"):
        _cc.create_shared(path + ".so", path)
        path += ".so"
    elif path.endswith(".tar"):
        tar_temp = _util.tempdir()
        _tar.untar(path, tar_temp.temp_dir)
        files = [tar_temp.relpath(x) for x in tar_temp.listdir()]
        _cc.create_shared(path + ".so", files)
        path += ".so"
    # Redirect to the load API
    return _LoadFromFile(path, fmt)

```

所以要存成 .tar 不然好像會出錯，可能是因爲如果用 so 就要使用 cc.create_share



# Export Libaray



```python
def export_library(self,
                       file_name,
                       fcompile=None,
                       **kwargs):
        """Export the module and its imported device code one library.
        This function only works on host llvm modules.
        It will pack all the imported modules
        Parameters
        ----------
        file_name : str
            The name of the shared library.
        fcompile : function(target, file_list, kwargs), optional
            Compilation function to use create dynamic library.
            If fcompile has attribute object_format, will compile host library
            to that format. Otherwise, will use default format "o".
        kwargs : dict, optional
            Additional arguments passed to fcompile
        """
        if self.type_key == "stackvm":
            if not file_name.endswith(".stackvm"):
                raise ValueError("Module[%s]: can only be saved as stackvm format."
                                 "did you build with LLVM enabled?" % self.type_key)
            self.save(file_name)
            return

        if not (self.type_key == "llvm" or self.type_key == "c"):
            raise ValueError("Module[%s]: Only llvm and c support export shared" % self.type_key)
        temp = _util.tempdir()
        if fcompile is not None and hasattr(fcompile, "object_format"):
            object_format = fcompile.object_format
        else:
            if self.type_key == "llvm":
                object_format = "o"
            else:
                assert self.type_key == "c"
                object_format = "cc"
        path_obj = temp.relpath("lib." + object_format)
        self.save(path_obj)
        files = [path_obj]
        is_system_lib = self.type_key == "llvm" and self.get_function("__tvm_is_system_module")()
        if self.imported_modules:
            path_cc = temp.relpath("devc.cc")
            with open(path_cc, "w") as f:
                f.write(_PackImportsToC(self, is_system_lib))
            files.append(path_cc)
        if not fcompile:
            if file_name.endswith(".tar"):
                fcompile = _tar.tar
            else:
                fcompile = _cc.create_shared
        fcompile(file_name, files, **kwargs)
```





# TVM JSON

NNVM JSON 和 mxnet 的 json 是不一樣的

# NNVM JSON

```python
https://docs.tvm.ai/dev/nnvm_json_spec.html?highlight=json
```



# 重要

這邊有 c++ load params 的方法

#Deploy NNVM Modules

https://docs.tvm.ai/deploy/nnvm.html#deploy-as-system-module

https://docs.tvm.ai/deploy/nnvm.html





#Load Data to GPU

```c++
int result = TVMArrayCopyFromBytes(x, fdata, 3 * 224 * 224 * 4);
```

這邊乘以 4 是因爲 float32 佔 4 個 byte，所以不管我們是用

```c++
float * fdata = new float [size];
```

還是

```c++
vector <float> fdata = vector <float> (size);
```

在 TVMArrayCopyFromBytes 都要乘以 4

# Get Data GPU

```c++
int sync =  TVMSynchronize (device_type, device_id, stvm);

int result = TVMArrayCopyToBytes(y, fdata.data(), 1 * 7 * 222 * 222 * 4);
```

從 GPU 讀取 Forward 後的資料到 CPU，一樣要乘以 4







# 進度

---

目前在 TX2 上使用基本的 TVM Compiled Model 不管是 CPU 或是 GPU 都已經可以正常使用了，不會像使用 MXNet 那樣發生問題

目前打算把 TVM 的一些基本 Function 打包成一個簡單的 class 以便之後調用

Pipe 是用來對特定一個 Data 在 TVMArray 和 Mat 或是 Float Array 之間轉換

例如我們常常會在輸入影像，這時我們就要把 Mat 轉成 Float Array 再轉成 TVMArray

影像輸入後，經過 Forward 後我們就要來看輸出的結果了，這時候除非我們是生成影像，如 GAN 等，不然我們應該都是轉成 Float Array 因爲我們可能要分類或是做 Detection 等



目前遇到了一個問題是 TVM 的 ssd 不支援 GPU Inference，測試後發現是 NMS 的部分，所以想說 NMS 的部分可能會改爲用 CPU 預測，Inception 的部分就保持 GPU，

NMS 選車

用 MXNet - CPU 

用 TVM - CPU

用 MXNet -GPU ，但是 C++ 版本還有問題



# Script

run.py : 基本的 tvm 用法

run_det_gpu : 沒有 nms 的部分的 ssd 

run_nms_cpu : 輸入是從 ssd 的 multi_feature_layer 出來的三個 symbol

會各自讀取 -det.json

和 nms-json

至於 nms-json 和 -det.json 的生成就要去 mxnet/example/ssd 用 split_ssd.py





TVM 記錄

每次 TVMCopy後最好還是使用 set_input 比較好

如果想要複製一個 TVMArray 到另一個 TVMArray

可以用 TVMArrayCopyFromTo

也可以直接用 = 這樣就是使用 Reference







