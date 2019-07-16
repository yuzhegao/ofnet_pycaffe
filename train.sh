#!/bin/bash
set -x

weights="../ofnet3/ResNet-50-model.caffemodel"
solver="solver.prototxt"
../../build/tools/caffe.bin train -solver="$solver" -weights="$weights" -gpu 1
