# TensorRT 4

由於目前 TX2 上只能跑 TenorRT 4，所以這就先以 TensorRT 4 爲主。

之前試過了很多個方法。









## Caffe + TensorRT

本來是想用這個組合的但是 Caffe - SSD 的部分一直有問題，雖然是可以編譯，但是不知怎麼的無法訓練，由於之後一定會需要自己再做訓練，所以弄了很久弄不好後就果斷放棄了，轉戰 Tensorflow + TensorRT



##Tensorflow + TensorRT

這個部分也嘗試了很多次才可以，原因是 TensorRT 的 Samples 使用的是 Tensorflow model zoo 裡的 Pretrained Model，但是那個 Github Project 的代碼有做過更動，所以 Frozen Graph (.pb) 的架構不太一樣，所以直接使用會出現問題



## Issue

錯誤提示 ：Reshape Dimension 的地方不能多個 -1

解決方法把

- `predictor/head/box_head.py`
- `ConvolutionalBoxHead.Predict()`

```python
box_encodings = tf.reshape(box_encodings,
					[1, -1, 1, self._box_code_size])
```



錯誤提示 : Concat Dimesion 必須一樣

解決辦法

修改 config.py，重新 Convert UFF 

```python
namespace_plugin_map = { 
    "MultipleGridAnchorGenerator": PriorBox,
    "Postprocessor": NMS,
    "Preprocessor": Input,
    # "ToFloat": Input,
    # "image_tensor": Input,
    #"MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
    "Concatenate/concat": concat_priorbox,
    "concat": concat_box_loc,
    "concat_1": concat_box_conf,
}

namespace_remove = { 
    "ToFloat",
    "image_tensor",
    "Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3",
}

def preprocess(dynamic_graph):
    # remove the unrelated or error layers
    dynamic_graph.remove(dynamic_graph.find_nodes_by_path(namespace_remove), remove_exclusive_dependencies=False)

    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)

    # Remove the Squeeze to avoid "Assertion `isPlugin(layerName)' failed"
    Squeeze = dynamic_graph.find_node_inputs_by_name(dynamic_graph.graph_outputs[0], 'Squeeze')
    dynamic_graph.forward_inputs(Squeeze)   
```

```python
#"MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
"Concatenate/concat": concat_priorbox,
```

這個部分應該是因爲 Graph 被修改過，新的節點叫做 Concatenate/concat，

詳情可以自己用 TensorBoard 觀察

----

# Tensorflow SSD

這個 Project 以前是用在資料夾 train 底下的 train.py 來做訓練，並且搭配每個以 .py 實現的 config，但是新版的是使用 model_main.py

```shell
python model_main.py --alsologtostderr \
--pipeline_config_path \samples/configs/ssd_inception_v2_coco.config \
--model_dir training
```

- model_dir : 訓練時所存的 checkpoint 和一些其他的資訊都會存在這裡



### 其他參數

除此之外 miuel_main.py 還有其他參數



#FineTune Pretrained Network With Detector Parameter

###Config Pipeline

這個是用來定義每個模型的訓練和測試用的參數

例如 mobilenetv2 可能就會有 coco 以及 pascal voc 兩種不同的 config

路徑在 `samples/config`



我們可以透過 `TensorBoard` 來觀察訓練的情況，預設 Terminal 是沒有輸出的

```shell
tensorboard --logdir training
```

# Workflow

## Tensorflow

```flow
train=>operation: Train the Model (model_main)
export=>operation: Export Frozen Graph (export_inefrence graph)
visualize=>operation: Visualize with Tensorboard (tensorboard)

train->export->visualize
```

### Export Frozen Graph

```shell
python export_inference_graph.py --pipeline_config_path samples/configs/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix mobilenet_v2_custom/model.ckpt-0 --output_directory mobilenet_v2_custom_frozen --input_shape '-1, 300, 300, 3'
```

###Visualize with Tensorboard

```shell
python ~/tf-tools/import_pb_to_tensorboard.py --model_dir frozen_inference_graph.pb --log_dir frozen.log
```





## TensorRT

```flow
copy=>operation: Copy Frozen Graph From Tensorflow
uff=>operation: Convert To Uff Representation
load=>operation: Load From C++
copy->uff->load

```

