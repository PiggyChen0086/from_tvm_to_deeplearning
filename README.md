# from_tvm_to_deeplearning
From TVM to Deep Learning: Cross-Framework-Device-One-Shot-Inference-Solution

## ./Benchmark

+ Contain the benchmark results for inference performance using MXNet and TVM respectively.

## ./Object_Detection_mxnet

+ ssd_hybridblock.ipynb  &emsp; --  &emsp; The program to build and train a SSD model by MXNet framework and export the model into files.
+ Reload_model.ipynb     &emsp; --  &emsp; The process to import the model from files and perform model inference by MXNet framework.
+ ssd_mxnet_to_tvm.ipynb &emsp; --  &emsp; The process to import the model from files and perform model inference by TVM stack.
+ ./mxnet_model          &emsp; --  &emsp; The exported files which includes the neural network's structure and weight.

## ./data

+ ./example_of_dataset_husky     &emsp; --  &emsp; The example of dataset using in the experiment of benchmark.
+ ./example_of_dataset_tigercat  &emsp; --  &emsp; The dataset for validation in the experiment of benchmark.
+ ./pikachu                      &emsp; --  &emsp; The training dataset containing the characters about Pikachu for object detection model. 
 
## ./pre-trained_model

+ deployment_mxnet.py  &emsp; --  &emsp; Tutorials for using MXNet framework in pre-trained model's deployment.
+ deployment_tvm.py    &emsp; --  &emsp; Tutorials for using TVM stack in pre-trained model's deployment.
+ mxnet_vs_tvm.ipynb   &emsp; --  &emsp; The comparison of MXNet and TVM in model inference.
+ ./ToolKit            &emsp; --  &emsp; A demo for deep learning model's cross-device deployment and the toolkits containing the functions for loading test data, pre-processing and post-processing for data, establishing sftp connections to transfer model's exported file.
