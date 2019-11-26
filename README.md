# glDelegateBench
quick and dirty inference time benchmark for TFLite gles delegate

The TensorFlow team announced TFLite GPU delegate and published related docs [2][3] in Jan 2019. But except Mobilenet V1 classifier, there is no publicly available app to evaluate it, so I wrote a quick and dirty app to evaluate other models.

For the 4 public models mentioned in [1], I got the following numbers on Pixel 2.

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 150 | 75 | 21 |
|PoseNet   | 183 | 96 | 40 |
|DeepLab V3| 219 | 131 | 91 |
|Mobilenet SSD V2 COCO| 264 | 158 | 49 |

On Xiaomi Mi 9, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 39 | 35 | 15 |
|PoseNet   | 48 | 47 | 19 |
|DeepLab V3| 61 | 64 | 65 |
|Mobilenet SSD V2 COCO| 69 | 75 | 36 |

On Pixel 3a, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 113 | 80 | 52 |
|PoseNet   | 138 | 96 | 78 |
|DeepLab V3| 173 | 132 | 144 |
|Mobilenet SSD V2 COCO| 200 | 167 | 113 |


Check https://github.com/freedomtan/glDelegateBenchmark/ for iOS code

## add a `local_tflite_aar` branch to test ruy, the new TFLite CPU backend
### on Pixel 2, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 117 | 37 | 20 |
|PoseNet   | 140 | 47 | 39 |
|DeepLab V3| 177 | 72 | 122 |
|Mobilenet SSD V2 COCO| 202 | 75 | 60 |

### on Pixel 3a, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 107 | 44 | 51 |
|PoseNet   | 131 | 57 | 77 |
|DeepLab V3| 164 | 82 | 145 |
|Mobilenet SSD V2 COCO| 184 | 86 | 113 |


## Update Oct 31, 2019. Nightly aar binaries are with ruy and OpenCL backend
### on Pixel 2 (w/ libOpenCL-pixel.so from Pixel 3), I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU OpenCL (ms)|GPU GL Compute Shader (ms)|
|----------|------------:|-------------:|---:|---:|
|Mobilenet | 118 | 34 | 10 | 21 |
|PoseNet   | 142 | 43 | 14 | 41 |
|DeepLab V3| 174 | 75 | 21 | 69 |
|Mobilenet SSD V2 COCO| 202 | 73 | 18 | 48 |

### on Pixel 3a, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU (ms)|
|----------|------------:|-------------:|---:|
|Mobilenet | 107 | 44 | 30 |
|PoseNet   | 131 | 57 | 42 |
|DeepLab V3| 164 | 82 | 66 |
|Mobilenet SSD V2 COCO| 184 | 86 | 55 |

### on Pixel 4, I got

|model name|CPU 1 thread (ms)|CPU 4 threads (ms) |GPU Delegate (ms)| NNAPI (ms)|
|----------|------------:|-------------:|---:|---:|
|Mobilenet | 42 | 13 | 8 | 7 | 
|PoseNet   | 52 | 15 | 11 | 11 |
|DeepLab V3| 66 | 25 | 20 | 98 | 
|Mobilenet SSD V2 COCO| 70 | 24 | 16 | 86 | 

[1] https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7

[2] https://www.tensorflow.org/lite/performance/gpu

[3] https://www.tensorflow.org/lite/performance/gpu_advanced
