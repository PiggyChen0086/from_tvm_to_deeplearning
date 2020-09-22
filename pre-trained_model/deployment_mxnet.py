#!/usr/bin/python
# -*- coding: UTF-8 -*-

# terminal: $ python3 mxnet.py -p test_data_path -t target

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

img_list = []


def transform_image_np(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    rgb_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    rgb_std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    return (image.astype('float32') / 255 - rgb_mean) / rgb_std
    

def get_test_data(test_data_path, max):
    # test data
    # test_data_path = "/Users/Mac_Chen/Desktop/TVM/from_tvm_to_deeplearning/pre-trained_model/tiger_cat"
    count = 0

    img_paths = glob.glob(os.path.join(test_data_path, "*.JPEG"))
    img_paths.sort()

    for path in img_paths:
        count += 1
        if count > max:
            break
    
        # Images under image files, either single channel (Black and White) or triple channel (RGB).
        # Unify the format of read images to prevent “ValueError: axes don't match array ”
        np_image = Image.open(path).convert("RGB").resize((256, 256))
        np_img = np_image.crop((16, 16, 240, 240))
        img_list.append(np.array(np_img))
    #     plt.imshow(np.array(np_img))
    #     plt.show()


# def usage():
#     print('Usage: \npython3 from_mxnet.py [-h] [-p] [-t] [-m]')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Test data path')
    parser.add_argument('-t', '--target', required=True, help='Target device for inference')
    parser.add_argument('-m', '--max', type=int, default=100, help='Retrieve the maximum number of images')
    args = parser.parse_args()
    
    argv_test_data_path = args.path
    argv_target = args.target
    argv_max = args.max
    
    print(argv_test_data_path)
    print(argv_target)
    print(argv_max)
    
    print('Download pre-trained model from model_zoo')
    # download pre-trained model from mxnet model_zoo
    block = vision.get_model('MobileNet1.0', pretrained=True)
    
    # ImageNet Label
    # Synset for converting the number of ImageNet classes to human vocabulary
    synset_path = "./imagenet1000_clsid_to_human.txt"

    with open(synset_path) as f:
        # text_labels = [' '.join(l.split()[1:]) for l in f]
        text_labels = eval(f.read())
        
    get_test_data(argv_test_data_path, argv_max)
    
    start_time_mxnet = time.time()
    prob_avg = 0
    count = 0
    for img in img_list:
        pred = block(nd.array(transform_image_np(img)))
        prob = mx.nd.softmax(pred)[0].asnumpy()

        idx = mx.nd.topk(pred, k=1)[0].astype('int').asnumpy().tolist()
    #     print('With prob = %.5f, it contains %s' % (prob[idx[0]], text_labels[idx[0]]))
        prob_avg += prob[idx[0]]
        count += 1

    print('Average accuracy = %0.5f' % float(prob_avg / count))
    print('Cost of time: %.5f sec' % (time.time() - start_time_mxnet))


    

if __name__ == '__main__':
    main(sys.argv[1:])