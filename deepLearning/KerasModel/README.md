1. LeNet-5 = CNN
/usr/local/lib/python3.5/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Reshape (Reshape)            (None, 28, 28, 1)         0         
_________________________________________________________________
Conv_1 (Conv2D)              (None, 28, 28, 32)        832       
_________________________________________________________________
MaxPool_1 (MaxPooling2D)     (None, 14, 14, 32)        0         
_________________________________________________________________
Conv_2 (Conv2D)              (None, 14, 14, 64)        51264     
_________________________________________________________________
MaxPool_2 (MaxPooling2D)     (None, 7, 7, 64)          0         
_________________________________________________________________
Flattern (Flatten)           (None, 3136)              0         
_________________________________________________________________
Dense_1 (Dense)              (None, 1024)              3212288   
_________________________________________________________________
DropOut_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
Soft_max_1 (Dense)           (None, 10)                10250     
=================================================================
Total params: 3,274,634
Trainable params: 3,274,634
Non-trainable params: 0
_________________________________________________________________
None
WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.
Train on 48000 samples, validate on 12000 samples
2018-04-17 13:27:05.437722: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-04-17 13:27:05.521175: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-04-17 13:27:05.521623: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1344] Found device 0 with properties: 
name: GeForce 940M major: 5 minor: 0 memoryClockRate(GHz): 1.176
pciBusID: 0000:01:00.0
totalMemory: 1.96GiB freeMemory: 1.55GiB
2018-04-17 13:27:05.521650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1423] Adding visible gpu devices: 0
2018-04-17 13:27:06.036475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-04-17 13:27:06.036514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:917]      0 
2018-04-17 13:27:06.036521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 0:   N 
2018-04-17 13:27:06.036693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1307 MB memory) -> physical GPU (device: 0, name: GeForce 940M, pci bus id: 0000:01:00.0, compute capability: 5.0)
Epoch 1/20
48000/48000 [==============================] - 47s 977us/step - loss: 0.9378 - acc: 0.7789 - val_loss: 0.3937 - val_acc: 0.9351
Epoch 2/20
48000/48000 [==============================] - 45s 934us/step - loss: 0.3915 - acc: 0.9301 - val_loss: 0.3142 - val_acc: 0.9567
Epoch 3/20
48000/48000 [==============================] - 48s 997us/step - loss: 0.3205 - acc: 0.9507 - val_loss: 0.2766 - val_acc: 0.9655
Epoch 4/20
48000/48000 [==============================] - 46s 948us/step - loss: 0.2844 - acc: 0.9611 - val_loss: 0.2571 - val_acc: 0.9696
Epoch 5/20
48000/48000 [==============================] - 44s 922us/step - loss: 0.2627 - acc: 0.9679 - val_loss: 0.2419 - val_acc: 0.9745
Epoch 6/20
48000/48000 [==============================] - 43s 899us/step - loss: 0.2479 - acc: 0.9723 - val_loss: 0.2329 - val_acc: 0.9762
Epoch 7/20
48000/48000 [==============================] - 44s 914us/step - loss: 0.2360 - acc: 0.9751 - val_loss: 0.2257 - val_acc: 0.9781
Epoch 8/20
48000/48000 [==============================] - 44s 919us/step - loss: 0.2282 - acc: 0.9774 - val_loss: 0.2197 - val_acc: 0.9798
Epoch 9/20
48000/48000 [==============================] - 46s 950us/step - loss: 0.2216 - acc: 0.9788 - val_loss: 0.2163 - val_acc: 0.9811
Epoch 10/20
48000/48000 [==============================] - 43s 904us/step - loss: 0.2151 - acc: 0.9808 - val_loss: 0.2136 - val_acc: 0.9812
Epoch 11/20
48000/48000 [==============================] - 43s 904us/step - loss: 0.2091 - acc: 0.9822 - val_loss: 0.2131 - val_acc: 0.9821
Epoch 12/20
48000/48000 [==============================] - 43s 892us/step - loss: 0.2061 - acc: 0.9827 - val_loss: 0.2052 - val_acc: 0.9828
Epoch 13/20
48000/48000 [==============================] - 42s 881us/step - loss: 0.2017 - acc: 0.9841 - val_loss: 0.2012 - val_acc: 0.9848
Epoch 14/20
48000/48000 [==============================] - 43s 892us/step - loss: 0.1976 - acc: 0.9851 - val_loss: 0.2001 - val_acc: 0.9843
Epoch 15/20
48000/48000 [==============================] - 43s 905us/step - loss: 0.1941 - acc: 0.9863 - val_loss: 0.1982 - val_acc: 0.9854
Epoch 16/20
48000/48000 [==============================] - 45s 928us/step - loss: 0.1921 - acc: 0.9864 - val_loss: 0.1961 - val_acc: 0.9852
Epoch 17/20
48000/48000 [==============================] - 44s 924us/step - loss: 0.1888 - acc: 0.9869 - val_loss: 0.1940 - val_acc: 0.9851
Epoch 18/20
48000/48000 [==============================] - 43s 893us/step - loss: 0.1859 - acc: 0.9874 - val_loss: 0.1919 - val_acc: 0.9860
Epoch 19/20
48000/48000 [==============================] - 43s 896us/step - loss: 0.1825 - acc: 0.9888 - val_loss: 0.1897 - val_acc: 0.9863
Epoch 20/20
48000/48000 [==============================] - 44s 908us/step - loss: 0.1808 - acc: 0.9889 - val_loss: 0.1879 - val_acc: 0.9867
10000/10000 [==============================] - 2s 182us/step
[0.17715083248615265, 0.9885]

