# 進度

------



# 注意

需要注意的是 flt 和 這個 Project 裡的一些 NDArray 到 Opencv 之間的轉換有稍微不同

#TX2

- [ ] C++ Version

## Todo

- [ ] 計劃把兩個 Repos 合在一起

---

## SSD

目前已經可以預測了，只差測試速度有沒有比 MXNet 版本還快就行了。

### CUDNN

由於 TVM 也能直接使用 CUDNN 作爲 Library，但是實驗過後加上與 Community 討論後發現，TVM 的 CUDNN 在 depth-wise Conv 還有些問題，但這不代表直接使用 TVM 進行 Operator Tuning 也還會有這些問題，所以目前還在嘗試對 SSD - Inceptionv3-512 進行 Tuning

---

## YOLO

目前以經可以 Tune 像是 TVM Tutorial 的 Yolov3 了，不管是原始大小

608, 416, 320 都可以

不過 Tiny-Yolo 的部分就不行了，直接使用 tiny-yolo.cfg 來做 nnvm.compile 會失敗

### C++

Yolov3 的 C++ 執行碼也還差轉換成 detection 的部分，



| Model                   | Dataset | Size | Device | Time  |
| ----------------------- | ------- | ---- | ------ | ----- |
| Inception_v3 + mxnet    | voc     | 300  | 1060   | 15 ms |
| Inception_v3 + tensorrt | voc     | 300  | 1060   | 11 ms |
| MobileNet_v2 + mxnet    | coco    | 300  | 1060   | 7 ms  |
| MobileNet_v2 + tensorrt | coco    | 300  | 1060   | 5 ms  |
| Inception_v3 + tensorrt | voc     | 300  | tx2    | 58 ms |
| Inception_v3 + mxnet    | voc     | 300  | tx2    | 75 ms |
| MobileNet_v2 + mxnet    | voc     | 300  | tx2    | 33 ms |


##inception v3, 300 , voc inference 3 times on tx2f

Time taken for inference is 104.249 ms.
Time taken for inference is 50.8974 ms.
Time taken for inference is 47.7279 ms.

# mobilenet v2, 300, coco, inference 3 times on tx2

Time taken for inference is 126.487 ms.
Time taken for inference is 54.8379 ms.
Time taken for inference is 38.48 ms.



# mobilenet v2, 300, voc                     tx2

Time taken for inference is 158.336 ms.
Time taken for inference is 55.8823 ms.
Time taken for inference is 86.5122 ms.
Time taken for inference is 80.4371 ms.
Time taken for inference is 78.7081 ms.
Time taken for inference is 44.8208 ms.
Time taken for inference is 55.4329 ms.
Time taken for inference is 39.7148 ms.
Time taken for inference is 37.8969 ms.
Time taken for inference is 50.7908 ms.



fobilenet v2, 300, coco, inference 3 times on tx2, batch size 2





# inceptiob_v2 batchsize = 2 , size =300, 1060

> Time taken for inference is 14.6986 ms.                                                                                                                                                                            
> Time taken for inference is 14.5524 ms.
> Time taken for inference is 14.8162 ms.                                                                                                                                                                            
> Time taken for inference is 14.5819 ms.
> Time taken for inference is 14.5935 ms.
> Time taken for inference is 14.6019 ms.
> Time taken for inference is 14.582 ms.
> Time taken for inference is 14.658 ms.
> Time taken for inference is 14.2659 ms.
> Time taken for inference is 13.66 ms.
> Time taken for inference is 13.6689 ms.
> Time taken for inference is 13.631 ms.
> Time taken for inference is 13.7081 ms.
> Time taken for inference is 13.7246 ms.
> Time taken for inference is 13.7466 ms.
> Time taken for inference is 13.6691 ms.
> Time taken for inference is 13.8347 ms.
> Time taken for inference is 13.7015 ms.
> Time taken for inference is 13.6591 ms.
> Time taken for inference is 15.2436 ms.

#inceptiob_v2 batchsize = 1 , size =300 1060

Time taken for inference is 9.27177 ms.                                                                                                                                                                            
Time taken for inference is 9.29634 ms.                                                                                                                                                                            
Time taken for inference is 9.27577 ms.                                                                                                                                                                            
Time taken for inference is 9.21098 ms.                                                                                                                                                                            
Time taken for inference is 9.22367 ms.                                                                                                                                                                            
Time taken for inference is 9.2305 ms.                                                                                                                                                                             
Time taken for inference is 9.27467 ms.                                                                                                                                                                            
Time taken for inference is 9.31359 ms.                                                                                                                                                                            
Time taken for inference is 9.42078 ms.                                                                                                                                                                            
Time taken for inference is 9.41277 ms.                                                                                                                                                                            
Time taken for inference is 9.39275 ms.                                                                                                                                                                            
Time taken for inference is 9.23167 ms.                                                                                                                                                                            
Time taken for inference is 9.27282 ms.                                                                                                                                                                            
Time taken for inference is 9.37258 ms.                                                                                                                                                                            
Time taken for inference is 9.31664 ms.                                                                                                                                                                            
Time taken for inference is 8.76294 ms.                                                                                                                                                                            
Time taken for inference is 8.71558 ms.                                                                                                                                                                            
Time taken for inference is 8.72915 ms.                                                                                                                                                                            
Time taken for inference is 8.71069 ms.                                                                                                                                                                            
Time taken for inference is 8.72927 ms.



# inception v2 1060 batch size =2 

Time taken for inference is 13.6954 ms.
Time taken for inference is 13.714 ms.
Time taken for inference is 14.322 ms.
Time taken for inference is 14.2064 ms.
Time taken for inference is 13.705 ms.
Time taken for inference is 13.6779 ms.
Time taken for inference is 13.6905 ms.
Time taken for inference is 13.6972 ms.
Time taken for inference is 13.6474 ms.
Time taken for inference is 13.6567 ms.
Time taken for inference is 13.6838 ms.
Time taken for inference is 13.6752 ms.
Time taken for inference is 13.6941 ms.
Time taken for inference is 13.7213 ms.
Time taken for inference is 13.6902 ms.
Time taken for inference is 13.92 ms.
Time taken for inference is 13.7524 ms.
Time taken for inference is 13.7871 ms.
Time taken for inference is 13.7253 ms.
Time taken for inference is 13.7269 ms.

#inception v2 tx2 300

cp inception_v2_custom.pb.uff ~/trt/data/ssd/sample_ssd.uff
➜  ssd ./ssd
../../../../data/ssd/sample_ssd.uff
Begin parsing model...
End parsing model...
Begin building engine...
End building engine...
 Num batches  1
 Data Size  270000
*** deserializing
Time taken for inference is 191.916 ms.
Time taken for inference is 68.7757 ms.
Time taken for inference is 71.9676 ms.
Time taken for inference is 102.865 ms.
Time taken for inference is 56.7111 ms.
Time taken for inference is 54.2524 ms.
Time taken for inference is 70.9255 ms.
Time taken for inference is 82.5483 ms.
Time taken for inference is 51.9699 ms.
Time taken for inference is 62.0943 ms.
Time taken for inference is 42.3137 ms.
Time taken for inference is 41.9262 ms.
Time taken for inference is 40.1295 ms.
Time taken for inference is 40.343 ms.
Time taken for inference is 45.3226 ms.
Time taken for inference is 48.8774 ms.
Time taken for inference is 52.9085 ms.
Time taken for inference is 53.1684 ms.
Time taken for inference is 57.4797 ms.
Time taken for inference is 41.9189 ms.





# Inception v2  tx2 300 Max N Mode

➜  ssd sudo nvpmodel -m 0
[sudo] password for nvidia:
➜  ssd ./bin/sample_uff_ssd
➜  ssd ./ssd
../../../../data/ssd/sample_ssd.uff
Begin parsing model...
End parsing model...
Begin building engine...
1End building engine...
 Num batches  1
 Data Size  270000
*** deserializing
Time taken for inference is 195.813 ms.
Time taken for inference is 61.198 ms.
Time taken for inference is 40.8578 ms.
Time taken for inference is 41.356 ms.
Time taken for inference is 56.7648 ms.
Time taken for inference is 61.2029 ms.
Time taken for inference is 40.0289 ms.
Time taken for inference is 41.1674 ms.
Time taken for inference is 55.9448 ms.
Time taken for inference is 74.8671 ms.
Time taken for inference is 70.7947 ms.
Time taken for inference is 47.6523 ms.
Time taken for inference is 50.841 ms.
Time taken for inference is 47.7755 ms.
Time taken for inference is 37.8256 ms.
Time taken for inference is 35.4132 ms.
Time taken for inference is 35.3722 ms.
Time taken for inference is 35.565 ms.
Time taken for inference is 35.1672 ms.
Time taken for inference is 35.4236 ms.



##Inception v2  tx2 300 Max N Mode Batch Size = 2

Time taken for inference is 434.407 ms.
Time taken for inference is 192.524 ms.
Time taken for inference is 169.773 ms.
Time taken for inference is 90.1105 ms.
Time taken for inference is 111.962 ms.
Time taken for inference is 80.1043 ms.
Time taken for inference is 99.0463 ms.
Time taken for inference is 78.0592 ms.
Time taken for inference is 68.1463 ms.
Time taken for inference is 67.7724 ms.
Time taken for inference is 65.263 ms.
Time taken for inference is 61.7148 ms.
Time taken for inference is 72.3871 ms.
Time taken for inference is 72.4559 ms.
Time taken for inference is 72.7227 ms.
Time taken for inference is 69.0078 ms.
Time taken for inference is 61.7938 ms.
Time taken for inference is 61.722 ms.
Time taken for inference is 62.0578 ms.
Time taken for inference is 61.5702 ms.