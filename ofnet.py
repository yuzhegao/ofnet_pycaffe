import sys
sys.path.append('python')

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L, params as P

## False if TRAIN, True if TEST
bn_global_stats = False


def _conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    '''Helper to build a conv -> BN -> relu block.
    '''
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    return conv, bn, scale

def _conv_bn_scale_relu(bottom, nout, bias_term=False, **kwargs):
    global bn_global_stats
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu

def _deconv_bn_scale_relu(bottom, nout, kernel_size, stride, pad, bias_term=False):
    ## just a bilinear upsample (lr_mult=0)
    global bn_global_stats

    conv = L.Deconvolution(bottom,
        convolution_param=dict(num_output=nout, kernel_size=kernel_size, stride=stride, pad=pad, bias_term=bias_term,
        weight_filler={"type": "bilinear"}), param=[dict(lr_mult=0)])
    bn = L.BatchNorm(conv, use_global_stats=bn_global_stats, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    out_relu = L.ReLU(scale, in_place=True)

    return conv, bn, scale, out_relu

def _conv_relu(bottom, nout, bias_term=False, **kwargs):
    conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,**kwargs)
    out_relu = L.ReLU(conv, in_place=True)

    return out_relu, conv



def _aspp_block(n, bottom, dil_rates=[1,6,12,18], out_dim=8):
    n.relu_aspp_1, n.aspp1 = _conv_relu(bottom, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})
    n.relu_6_1, n.conv6_1 = _conv_relu(n.relu_aspp_1, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_2, n.aspp_2 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[0], pad=dil_rates[0],
                                         weight_filler={"type": "msra"})
    n.relu_6_2, n.conv6_2 = _conv_relu(n.relu_aspp_2, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_3, n.aspp_3 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[1], pad=dil_rates[1],
                                         weight_filler={"type": "msra"})
    n.relu_6_3, n.conv6_3 = _conv_relu(n.relu_aspp_3, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    n.relu_aspp_4, n.aspp_4 = _conv_relu(bottom, nout=out_dim, kernel_size=3, dilation=dil_rates[2], pad=dil_rates[2],
                                         weight_filler={"type": "msra"})
    n.relu_6_4, n.conv6_4 = _conv_relu(n.relu_aspp_4, nout=out_dim, kernel_size=1, weight_filler={"type": "msra"})

    concat_layers = [n.relu_6_1, n.relu_6_2, n.relu_6_3, n.relu_6_4]
    n.aspp_concat = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))


def output_branch(n, bottom, task):
    """ For edge and ori output (task: edge or ori)
        edge and ori has same branch [8,8,8,8,4,1]
    """
    conv = 'unet1a_plain1a_conv_{}'.format(task)
    bn = 'unet1a_plain1a_bn_{}'.format(task)
    scale = 'unet1a_plain1a_scale_{}'.format(task)
    relu1 = 'unet1a_plain1a_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu1] = _conv_bn_scale_relu(bottom, nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1a_plain1b_conv_{}'.format(task)
    bn = 'unet1a_plain1b_bn_{}'.format(task)
    scale = 'unet1a_plain1b_scale_{}'.format(task)
    relu2 = 'unet1a_plain1b_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu2] = _conv_bn_scale_relu(n[relu1], nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1a_plain1c_conv_{}'.format(task)
    bn = 'unet1a_plain1c_bn_{}'.format(task)
    scale = 'unet1a_plain1c_scale_{}'.format(task)
    relu3 = 'unet1a_plain1c_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu3] = _conv_bn_scale_relu(n[relu2], nout=8, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_plain1a_conv_{}'.format(task)
    bn = 'unet1b_plain1a_bn_{}'.format(task)
    scale = 'unet1b_plain1a_scale_{}'.format(task)
    relu4 = 'unet1b_plain1a_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu4] = _conv_bn_scale_relu(n[relu3], nout=8, bias_term=True, kernel_size=3,
                                                            stride=1, pad=1, weight_filler={"type": "msra"},
                                                            bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_plain1b_conv_{}'.format(task)
    bn = 'unet1b_plain1b_bn_{}'.format(task)
    scale = 'unet1b_plain1b_scale_{}'.format(task)
    relu5 = 'unet1b_plain1b_relu_{}'.format(task)
    n[conv], n[bn], n[scale], n[relu5] = _conv_bn_scale_relu(n[relu4], nout=4, bias_term=True, kernel_size=3,
                                                             stride=1, pad=1, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})

    conv = 'unet1b_{}'.format(task)
    n[conv] = L.Convolution(n[relu5], num_output=1, bias_term=True, kernel_size=1,
                                                             stride=1, pad=0, weight_filler={"type": "msra"},
                                                             bias_filler={"type": "constant", "value": 0.0})


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

def _resnet_dilation_block(name, n, bottom, nout, branch1=False, initial_stride=2, dil_rate=2):
    '''Basic ResNet block.
    '''
    if branch1:
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride,
            pad=0, weight_filler={"type": "msra"})
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
    # dilation
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=dil_rate, weight_filler={"type": "msra"}, dilation=dil_rate)
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


def resnet50_branch(n, bottom):
    '''ResNet 50 layers.
    '''
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        bottom, 64, bias_term=True, kernel_size=7, pad=3, stride=2, weight_filler={"type": "msra"})
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    # stage 2
    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64) # res2c_relu

    # stage 3
    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128) # res3d_relu

    # stage 4
    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256) # res4f_relu

    # stage 5
    _resnet_dilation_block('5a', n, n.res4f_relu, 512, branch1=True,initial_stride=1)
    _resnet_dilation_block('5b', n, n.res5a_relu, 512)
    _resnet_dilation_block('5c', n, n.res5b_relu, 512) # res5c_relu


def decoder_path(n, bottom):
    n.unet3a_deconv_up, n.unet3a_bn3a_deconv, n.unet3a_scale3a_deconv, n.unet3a_deconv_relu = \
        _deconv_bn_scale_relu(bottom, nout=256, kernel_size=7, stride=4, pad=0)

    crop_bottoms = [n.unet3a_deconv_relu, n.res2c_relu]
    n.unet3a_crop = L.Crop(*crop_bottoms, crop_param={'axis': 2, 'offset': 1})  ## crop 1/4

    concat_layers = [n.unet3a_crop, n.res2c_relu]
    n.unet3a_concat = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))  ## concat with res2c

    _resnet_block('6a', n, n.unet3a_concat, 128)
    _resnet_block('6b', n, n.res6a_relu, 4, branch1=True,initial_stride=1)  ## res 6

    n.unet1a_deconv_up, n.unet1a_bn1a_deconv, n.unet1a_scale1a_deconv, n.unet1a_deconv_relu = \
        _deconv_bn_scale_relu(n.res6b_relu, nout=16, kernel_size=7, stride=4, pad=0)  ## deconv

    crop_bottoms = [n.unet1a_deconv_relu, n.data]
    n.unet1a_crop = L.Crop(*crop_bottoms, crop_param={'axis': 2, 'offset': 1})  ## crop 1 -> occlusion cue

def edge_path(n, bottom):
    n.res5c_1, n.res5c_1_bn, n.res5c_1_scale, n.res5c_1_relu = \
        _conv_bn_scale_relu(bottom, nout=16, bias_term=False, kernel_size=1,
                            stride=1, pad=0, weight_filler={"type": "msra"})

    n.res5c_up1, n.res5c_up1_bn, n.res5c_up1_scale, n.res5c_up1_relu = \
        _deconv_bn_scale_relu(n.res5c_1_relu, nout=16, kernel_size=8, stride=4, pad=0)

    crop_bottoms = [n.res5c_up1_relu, n.res2c_relu]
    n.res5c_crop1 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1/4

    n.res5c_up2, n.res5c_up2_bn, n.res5c_up2_scale, n.res5c_up2_relu = \
        _deconv_bn_scale_relu(n.res5c_crop1, nout=16, kernel_size=8, stride=4, pad=0)

    crop_bottoms = [n.res5c_up2_relu, n.data]
    n.res5c_crop2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1 -> from res backbonb

    n.unet1a_conv1_edge, n.unet1a_bn_conv1_edge, n.unet1a_scale_conv1_edge, n.unet1a_conv1_relu_edge = \
        _conv_bn_scale_relu(n.data, nout=8, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet1a_conv2_edge, n.unet1a_bn_conv2_edge, n.unet1a_scale_conv2_edge, n.unet1a_conv2_relu_edge = \
        _conv_bn_scale_relu(n.unet1a_conv1_relu_edge, nout=4, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})

    n.unet1a_conv3_edge, n.unet1a_bn_conv3_edge, n.unet1a_scale_conv3_edge, n.unet1a_conv3_relu_edge = \
        _conv_bn_scale_relu(n.unet1a_conv2_relu_edge, nout=16, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})  ## 1 -> from data directly

def ori_path(n, bottom):
    _aspp_block(n, bottom)  ## aspp module

    n.aspp_up1, n.aspp_up1_bn, n.aspp_up1_scale, n.aspp_up1_relu = \
        _deconv_bn_scale_relu(n.aspp_concat, nout=32, kernel_size=8, stride=4, pad=0)

    crop_bottoms = [n.aspp_up1_relu, n.res2c_relu]
    n.aspp_crop1 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1/4

    n.aspp_up2, n.aspp_up2_bn, n.aspp_up2_scale, n.aspp_up2_relu = \
        _deconv_bn_scale_relu(n.aspp_crop1, nout=16, kernel_size=8, stride=4, pad=0)

    crop_bottoms = [n.aspp_up2_relu, n.data]
    n.aspp_crop2 = L.Crop(*crop_bottoms, crop_param={'axis': 2})  ## crop 1


def ofnet(n,is_train=True):
    global bn_global_stats
    bn_global_stats = False if is_train else True

    resnet50_branch(n, n.data) ## resnet50 backbone

    ## plain 6
    n.plain6a_conv, n.plain6a_bn, n.plain6a_scale, n.plain6a_relu = \
            _conv_bn_scale_relu(n.res5c_relu, nout=256,bias_term=False, kernel_size=3,
            stride=1, pad=1, weight_filler={"type": "msra"})

    n.plain6b_conv, n.plain6b_bn, n.plain6b_scale, n.plain6b_relu = \
        _conv_bn_scale_relu(n.plain6a_relu, nout=256, bias_term=False, kernel_size=3,
                            stride=1, pad=1, weight_filler={"type": "msra"})
    ## plain6b_relu is the origin of three path -----

    ## decoder path
    decoder_path(n, n.plain6b_relu)

    ## edge path
    edge_path(n, n.plain6b_relu)

    ## ori path
    ori_path(n, n.plain6b_relu)


    ## edge output ----------------------
    concat_layers = [n.unet1a_crop, n.unet1a_conv3_relu_edge,n.res5c_crop2]
    n.unet1a_concat_edge = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))  ## concat with res2c

    output_branch(n, n.unet1a_concat_edge, task='edge')

    ## ori output -----------------------
    concat_layers = [n.unet1a_crop, n.aspp_crop2]
    n.unet1a_concat_ori = caffe.layers.Concat(*concat_layers, concat_param=dict(concat_dim=1))  ## concat with res2c

    output_branch(n, n.unet1a_concat_ori, task='ori')


if __name__ == '__main__':
    n = caffe.NetSpec()
    # n.data = L.DummyData(shape=[dict(dim=[1, 3, 224, 224])])
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])


    ofnet(n, is_train=True)
    with open('ofnet.prototxt', 'w') as f:
        f.write(str(n.to_proto()))

    net = caffe.Net('ofnet.prototxt',
                    '../ofnet3/ResNet-50-model.caffemodel',
                    caffe.TEST)









