# 進度

------



# 注意

需要注意的是 flt 和 這個 Project 裡的一些 NDArray 到 Opencv 之間的轉換有稍微不同

## TX2

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