'''Caffe ResNet NetSpec example.
Compatible with Kaiming He's pre-trained models.
https://github.com/KaimingHe/deep-residual-networks
'''
import numpy as np
import cv2

import sys
sys.path.append('python')

# Make sure that caffe is on the python path:
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
print('import caffe success')

import caffe
from caffe import layers as L, params as P


def _conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    '''Helper to build a conv -> BN -> relu block.
    '''
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale


def _resnet_block(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    if branch1:  ## if is branch1 in a stage, should change dim (like downsample in pytorch)
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
        ## don't forget param: weight_filler={"type": "xavier"},bias_filler={"type": "constant"},
        ## param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)  (if need)
    else:
        initial_stride = 1

    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0, weight_filler={"type": "msra"})
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1, weight_filler={"type": "msra"})
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)

    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0, weight_filler={"type": "msra"})
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)


def resnet50(n, bottom):
    '''ResNet 50 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=True, kernel_size=7, pad=3, stride=2, weight_filler={"type": "msra"})
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)

    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128)

    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256)

    _resnet_block('5a', n, n.res4f_relu, 512, branch1=True)
    _resnet_block('5b', n, n.res5a_relu, 512)
    _resnet_block('5c', n, n.res5b_relu, 512)

    n.pool5 = L.Pooling(
        n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)


def resnet101(n, bottom):
    '''ResNet 101 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=False, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)

    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    bottom = n.res3a_relu
    for i in range(3):
        _resnet_block('3b{}'.format(i + 1), n, bottom, 128)
        bottom = n['res3b{}_relu'.format(i + 1)]

    _resnet_block('4a', n, bottom, 256, branch1=True)
    bottom = n.res4a_relu
    for i in range(22):
        _resnet_block('4b{}'.format(i + 1), n, bottom, 256)
        bottom = n['res4b{}_relu'.format(i + 1)]

    _resnet_block('5a', n, bottom, 512, branch1=True)
    _resnet_block('5b', n, n.res5a_relu, 512)
    _resnet_block('5c', n, n.res5b_relu, 512)

    n.pool5 = L.Pooling(
        n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)


def resnet152(n, bottom):
    '''ResNet 152 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=False, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)

    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    bottom = n.res3a_relu
    for i in range(7):
        _resnet_block('3b{}'.format(i + 1), n, bottom, 128)
        bottom = n['res3b{}_relu'.format(i + 1)]

    _resnet_block('4a', n, bottom, 256, branch1=True)
    bottom = n.res4a_relu
    for i in range(35):
        _resnet_block('4b{}'.format(i + 1), n, bottom, 256)
        bottom = n['res4b{}_relu'.format(i + 1)]

    _resnet_block('5a', n, bottom, 512, branch1=True)
    _resnet_block('5b', n, n.res5a_relu, 512)
    _resnet_block('5c', n, n.res5b_relu, 512)

    n.pool5 = L.Pooling(
        n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)


def random_crop_and_flip(batch_data, padding_size):
    IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH = 224,224,3
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][:, x_offset:x_offset + IMG_HEIGHT,
                                y_offset:y_offset + IMG_WIDTH]

    return cropped_batch




if __name__ == '__main__':
    n = caffe.NetSpec()
    # n.data = L.DummyData(shape=[dict(dim=[1, 3, 224, 224])])
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
    resnet50(n, n.data)
    n.fc1000 = L.InnerProduct(n.pool5, num_output=1000)
    n.prob = L.Softmax(n.fc1000)

    """
    print(n['res2c'])
    print(n['res2c_relu'])
    print(n.res2c_relu)
    < caffe.net_spec.Top object at 0x7fce924bff50 >
    < caffe.net_spec.Top object at 0x7fce9244f090 >
    < caffe.net_spec.Top object at 0x7fce9244f090 >
    """
    layers = n.tops
    # for k,v in layers.items():
    #     print k

    with open('tmp.prototxt', 'w') as f:
        f.write(str(n.to_proto()))


    net = caffe.Net('tmp.prototxt',
                    '../ofnet3/ResNet-50-model.caffemodel',
                    caffe.TEST)
    # for b in net.blobs:
    #     print(b)

    ## feed data and run
    im = cv2.imread('2.jpg') # [H,W,3]
    im = im.astype(np.float32)
    # im -= np.array((104.00698793, 116.66876762, 122.67891434)) # so the mean rgb maybe useless?
    im = np.expand_dims(np.transpose(im,(2,0,1)),0).astype(np.float32)
    im = random_crop_and_flip(im,1)

    # dummy = np.random.rand(1,3,224,224).astype(np.float32)
    net.blobs['data'].data[...] = im
    net.forward()

    pred = net.blobs['prob'].data
    print(np.argmax(pred))

    """
    ## net.xxx and net.blobs[xxx] is different
    print(net.blobs['res2c'])
    print(net.res2c) # 'Net' object has no attribute 'res2c'
    """

    print(net.blobs['res2c'])
    print(net['res2c_relu']) # 'Net' object has no attribute '__getitem__'
    print(net.res2c)  # 'Net' object has no attribute 'res2c'









