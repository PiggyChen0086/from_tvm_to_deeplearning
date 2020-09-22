#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
import glob
import sys
from PIL import Image 

class DataProcessImageRecognition(object):
    def __init__(self):
        self.img_list = []
        self.count = 0
        
        # ImageNet Label
        # Synset for converting the number of ImageNet classes to human vocabulary
        synset_path = "./imagenet1000_clsid_to_human.txt"
        with open(synset_path) as f:
            self.text_labels = eval(f.read())
            

    def transform_image_np(self, image):
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        rgb_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        rgb_std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        return (image.astype('float32') / 255 - rgb_mean) / rgb_std
        
    
    def load_test_data(self, test_data_path, maxium=1):
        print('Load test data...')
        count = 0

        img_paths = glob.glob(os.path.join(test_data_path, "*.*[G|g]"))
        img_paths.sort()
        for path in img_paths:
            count += 1
            if count > maxium:
                break
    
            # Images under image files, either single channel (Black and White) or triple channel (RGB).
            # Unify the format of read images to prevent “ValueError: axes don't match array ”
            try:
                np_image = Image.open(path).convert("RGB").resize((256, 256))
            except:
                print('Read error in', path, ', skip')
                count -= 1
                pass
            else: 
                np_img = np_image.crop((16, 16, 240, 240))
                self.img_list.append(np.array(np_img))
        return self.img_list
            

    def post_process_image_recognition(self, output):
        self.count += 1
        print('In picture %d' % (self.count))
        tvm_output = output.asnumpy()[0]

        idx = np.argsort(tvm_output)[-3:][::-1]
        for i in range(3):
            print('With prob = %.5f, it contains %s' % (tvm_output[idx[i]], self.text_labels[idx[i]]))
    

    