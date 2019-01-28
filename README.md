# glDelegateBench
quick and dirty inference time benchmark for TFLite gles delegate

The TensorFlow team announced TFLite GPU delegate and published related docs [2][3] in Jan 2019. But except Mobilenet V1 classifier, there is no publicly available app to evaluate it, so I wrote a quick and dirty app to evaluate other models.

For the 4 public models mentioned in [1], I got the following numbers on Pixel 2.

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------|-------------|---|
|Mobilenet | 150 | 75 | 21 |
|PoseNet   | 183 | 96 | 40 |
|DeepLab V3| 219 | 131 | 01 |
|Mobilenet SSD V2 COCO| 264 | 158 | 49 |



[1] https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7

[2] https://www.tensorflow.org/lite/performance/gpu

[3] https://www.tensorflow.org/lite/performance/gpu_advanced
