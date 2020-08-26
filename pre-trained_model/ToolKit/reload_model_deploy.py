#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import argparse
import time

import numpy
import tvm
from tvm.contrib import graph_runtime 

from ImageRecognitionToolkit import DataProcessImageRecognition

img_list = []

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path of test data')
    parser.add_argument('--path', default='./', help='Path of model')
    parser.add_argument('--max', type=int, default=1, help='Retrieve the maximum number of images')
    args = parser.parse_args()
    
    argv_test_data = args.data
    argv_tmp_path = args.path
    argv_max = args.max
    
    tool_kit = DataProcessImageRecognition()
    img_list = tool_kit.load_test_data(argv_test_data, argv_max)
    
    print('Reload tvm model from file')
    loaded_lib = tvm.runtime.load_module(os.path.join(argv_tmp_path, "deploy_lib.so"))
    print("Tyep of lib: ", type(loaded_lib))

    print('Tvm: run the graph')
    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.GraphModule(loaded_lib['default'](ctx))
    
    print('Input the img')
    start_time_tvm = time.time()
    prob_avg = 0
    count = 0
    
    for img_ in img_list:
        m.set_input('data', tvm.nd.array(tool_kit.transform_image_np(img_).astype(dtype)))
        
        m.run()
        tvm_output = m.get_output(0)
        
        tool_kit.post_process_image_recognition(tvm_output)
    

    print('Cost of time: %.5f sec' % (time.time() - start_time_tvm))


if __name__ == '__main__':
    main(sys.argv[1:])

