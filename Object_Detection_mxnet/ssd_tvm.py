# !/usr/bin/python
# -*- coding: UTF-8 -*-

import mxnet as mx
from mxnet import contrib, gluon, image, nd, symbol
import tvm
from tvm import relay
from tvm.contrib import graph_runtime

from IPython import display
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import time, os


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0],
                         height=bbox[3]-bbox[1], fill=False, edgecolor=color,
                         linewidth=2)


def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    
    
def predict(tvm_output):
    anchors, cls_preds, bbox_preds = [nd.array(tvm_output.get_output(i).asnumpy()) for i in range(3)]
    print('Type anchors:{0}\nanchors:{1}\nType cls:{2}\ncls:{3}\nType bbox:{4}\nbbox:{5}'.format(type(anchors), anchors, type(cls_preds), cls_preds, type(bbox_preds), bbox_preds))
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]
    

def display_final(img, output, threshold):
    fig = plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    
    
def main():
    prefix_path = os.path.join('./', 'ssd_test')
    print(prefix_path)
    
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=prefix_path, epoch=0)
    print('Sym: ', sym)
    
    # load the target image
    set_figsize()
    img_ = image.imread('./pikachu.jpg')
    # plt.imshow(img_.asnumpy())
    feature = image.imresize(img_, 256, 256).astype('float32')
    input_ = feature.transpose((2, 0, 1)).expand_dims(axis=0)
    print('Image: {0}, type: {1}'.format(img_.shape, type(img_)))
    print('Input: {0}, type: {1}'.format(input_.shape, type(input_)))
    shape_dict = {'data': input_.shape}
    print('Shape_dict: ', shape_dict)
    
    # tvm
    # (mxnet.symbol)model --> relay.IRModule
    mod_, params_ = relay.frontend.from_mxnet(sym, shape_dict, arg_params=arg_params, aux_params=aux_params)
    print('Type of mod_: ', type(mod_))
    
    # relay.IRModule --> relay.graph
    target_ = tvm.target.create('llvm')
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod_, target_, params=params_)
        
    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input('data', tvm.nd.array(input_.astype(dtype).asnumpy()))
    m.set_input(**params)
    
    m.run()
    
    pict_output = predict(m)
    set_figsize((5, 5))
    display_final(img_, pict_output, threshold=0.3)
    
    
if __name__ == '__main__':
    main()