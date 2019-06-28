# 目前 MatVector_to_NDArray 的部分還有一些問題

雖然不知爲什麼


```python


	vector <float> ftotal = vector <float> (size);

	cv::Mat mat(h, w, CV_32FC3);

	for(int i = 0; i != b; i++){

		v[i].convertTo(mat, CV_32FC3);

		ftotal.assign(mat.begin <float> (), mat.end <float> ());

	}
	
	ndtotal.SyncCopyFromCPU(ftotal.data(), size);

```

這樣都程式碼可以使用，但是真正的問題應該是出在

NDArray 讀是 (3, 224, 224)

OpenCV 是 (224, 224, 3)

所以就算位置是連續的也不能直接讀取
