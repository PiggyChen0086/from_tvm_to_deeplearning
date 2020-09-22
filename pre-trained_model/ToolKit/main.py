#!/usr/bin/python
# -*- coding: UTF-8 -*-

# terminal: $ python3 from_mxnet.py -p test_data_path -t target

import mxnet as mx
from mxnet.gluon.model_zoo import vision
from mxnet import nd
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime

import numpy as np
# import matplotlib.pyplot as plt
import time
import os
import glob
from PIL import Image 
import argparse
import sys

from ImageRecognitionToolkit import DataProcessImageRecognition

img_list = []


# def export_model(graph, lib, params):
#     print("Export model...")
#     tmp_path = './'
#     local_lib_path = os.path.join(tmp_path, "deploy_lib.so")
#     local_graph_path = os.path.join(tmp_path, "deploy_graph.json")
#     local_params_path = os.path.join(tmp_path, "deploy_params.params")
#     
#     cross_compile =  "aarch64-unknown-linux-gnu-g++"
#     lib.export_library(local_lib_path, cc=cross_compile)
#     
#     # lib.export_library(local_lib_path)
# 
#     with open(local_graph_path, 'w') as fo:
#         fo.write(graph)
# 
#     with open(local_params_path, 'wb') as fo:
#         fo.write(relay.save_param_dict(params))
def export_model(lib):
    print("Export model...")
    tmp_path = './'
    local_lib_path = os.path.join(tmp_path, "deploy_lib.so")
    
    # cross_compile =  "aarch64-unknown-linux-gnu-g++"
    # lib.export_library(local_lib_path, cc=cross_compile)
    lib.export_library(local_lib_path)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Test data path')
    parser.add_argument('-t', '--target', required=True, help='Target device for inference')
    parser.add_argument('-m', '--max', type=int, default=1, help='Retrieve the maximum number of images')
    args = parser.parse_args()
    
    argv_test_data_path = args.path
    argv_target = args.target
    argv_max = args.max
    
    print(argv_test_data_path)
    print(argv_target)
    print(argv_max)
    
    # download pre-trained model from mxnet model_zoo
    block = vision.get_model('MobileNet1.0', pretrained=True)
    
    tool_kit = DataProcessImageRecognition()
    img_list = tool_kit.load_test_data(argv_test_data_path, argv_max)
    
    print('Relay: get model from mxnet...')
    img_ = tool_kit.transform_image_np(img_list[0])
    print('img', img_.shape, 'type: ', type(img_))

    shape_dict = {'data': img_.shape}
    print('Block: {0}, Dict_shape: {1}'.format(type(block), type(shape_dict)))

    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    print('Mod: {0}, Params: {1}'.format(type(mod), type(params)))
    func = mod['main']
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)


    print("Relay: build the graph")
    # target = 'llvm'
    if argv_target == 'llvm':
        target = 'llvm -mtriple=aarch64-linux-gnu -mattr=+neon'
        target_host = 'llvm -mtriple=x86_64-apple-darwin18.7.0'
        ctx = tvm.cpu(0)
    elif argv_target == 'cuda':
        target = tvm.target.create('cuda')
        target_host = tvm.target.create('cuda')
        ctx = tvm.gpu(0)
    else:
        target = argv_target
        
    with tvm.transform.PassContext(opt_level=3):
        # graph, lib, params = relay.build(func, target=target, params=params)  
        lib = relay.build(func, target=target, target_host=target_host, params=params) 
    # print("Graph: {0}, lib: {1}, params: {2}".format(type(graph), type(lib), type(params)))
    print("Lib: {0}".format(type(lib)))
    
    # export_model(graph, lib, params)
    export_model(lib)


    print('Tvm: run the graph')
    dtype = 'float32'
    # m = graph_runtime.create(graph, lib, ctx)
    m = graph_runtime.GraphModule(lib['default'](ctx))
    print('GraphModule: {0}'.format(type(m)))
    
    print('Input the img')
    start_time_tvm = time.time()
    prob_avg = 0
    count = 0
    
    for img_ in img_list:
        m.set_input('data', tvm.nd.array(tool_kit.transform_image_np(img_).astype(dtype)))
        # m.set_input(**params)

        m.run()
        tvm_output = m.get_output(0)
        print('Output: {0}'.format(type(tvm_output))) 
        
        tool_kit.post_process_image_recognition(tvm_output)
    

    print('Cost of time: %.5f sec' % (time.time() - start_time_tvm))
    

if __name__ == '__main__':
    main(sys.argv[1:])