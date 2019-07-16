import sys
sys.path.append('python')

caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
print('import caffe success')

import caffe
from caffe import layers as L, params as P

from ofnet import ofnet

def write_solver(base_lr=0.00003, iters=30000, snapshot='snapshot/ofnet'):
    """
    Generate solver.prototxt.
    base_lr: learning rate;
    iters: max iterations;
    snapshot: the prefix of saved models.
    """
    sovler_string = caffe.proto.caffe_pb2.SolverParameter() ## define solver

    sovler_string.net = 'ofnet.prototxt'
    sovler_string.test_iter.append(0)
    sovler_string.test_interval = 100000
    sovler_string.base_lr = base_lr
    sovler_string.lr_policy = 'step'
    sovler_string.gamma = 0.1
    sovler_string.iter_size = 3
    sovler_string.stepsize = 20000
    sovler_string.display = 20
    sovler_string.max_iter = iters
    sovler_string.momentum = 0.9
    sovler_string.weight_decay = 0.0002
    sovler_string.snapshot = 40000
    sovler_string.snapshot_prefix = snapshot

    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU

    # print str(sovler_string)
    print(sovler_string)
    exit()

    # with open('solver.prototxt', 'w') as f:
    #     f.write(str(sovler_string))

def net_forward():
    n = caffe.NetSpec()
    n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])


    ofnet(n, is_train=True)

    with open('ofnet_test.prototxt', 'w') as f:
        f.write(str(n.to_proto()))  ## write network

    net = caffe.Net('ofnet_test.prototxt',
                    '../ofnet3/ResNet-50-model.caffemodel',
                    caffe.TRAIN)


    import numpy as np
    dummy_im = np.random.rand(1, 3, 224, 224).astype(np.float32)

    net.blobs['data'].data[...] = dummy_im
    net.forward()

    pred = net.blobs['unet1b_ori'].data
    print(pred.shape)

    ## when use net_forward() to test the network, it's all right





def train(initmodel,gpu):
    # n = caffe.NetSpec()
    # # n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
    # n.data, n.label = L.ImageLabelmapData(include={'phase': 0}, ## 0-TRAIN 1-TEST
    #                                       image_data_param={
    #                                           'source': "../../data/PIOD/Augmentation/train_pair_320x320.lst",
    #                                           'batch_size': 5,
    #                                           'shuffle': True,
    #                                           'new_height': 0,
    #                                           'new_width': 0,
    #                                           'root_folder': "",
    #                                           'data_type': "h5"},
    #                                       transform_param={
    #                                           'mirror': False,
    #                                           'crop_size': 320,
    #                                           'mean_value': [104.006988525,116.668769836,122.678916931]
    #                                        },
    #                                       ntop=2)
    #
    # n.data, n.label = L.ImageLabelmapData(include={'phase': 1},  ## 0-TRAIN 1-TEST
    #                                 image_data_param={
    #                                     'source': "../../data/PIOD/Augmentation/train_pair_320x320.lst",
    #                                     'batch_size': 5,
    #                                     'shuffle': True,
    #                                     'new_height': 0,
    #                                     'new_width': 0,
    #                                     'root_folder': "",
    #                                     'data_type': "h5"},
    #                                 transform_param={
    #                                     'mirror': False,
    #                                     'crop_size': 320,
    #                                     'mean_value': [104.006988525, 116.668769836, 122.678916931]
    #                                 },
    #                                 ntop=2)
    #
    # n.label_edge, n.label_ori = L.Slice(n.label, slice_param={'slice_point': 1}, ntop=2)
    #
    # ofnet(n, is_train=True)
    #
    # loss_bottoms = [n.unet1b_edge,n.label_edge]
    # n.edge_loss = L.ClassBalancedSigmoidCrossEntropyAttentionLoss(*loss_bottoms,
    #                                  loss_weight=1.0, attention_loss_param={'beta': 4.0,'gamma': 0.5})
    #
    # loss_bottoms = [n.unet1b_ori, n.label_ori, n.label_edge]
    # n.ori_loss = L.OrientationSmoothL1Loss(*loss_bottoms, loss_weight=0.5, smooth_l1_loss_param={'sigma': 3.0})
    #
    # with open('ofnet.prototxt', 'w') as f:
    #     f.write(str(n.to_proto())) ## write network

    # net = caffe.Net('ofnet.prototxt',
    #                 '../ofnet3/ResNet-50-model.caffemodel',
    #                 caffe.TRAIN)

    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    # write_solver()
    solver = caffe.SGDSolver('solver.prototxt')
    if initmodel:
        solver.net.copy_from(initmodel)  ## why use solver.net.copy_from() to load  pretrained model?

    solver.step(solver.param.max_iter)



if __name__ == '__main__':
    # train(initmodel='../ofnet3/ResNet-50-model.caffemodel', gpu=0)
    # net_forward()
    write_solver()