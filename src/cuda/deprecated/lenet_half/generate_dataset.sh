#!/bin/bash

set -x
set -e

MNIST_DATA_DIR=caffe/data/mnist
MNIST_DIR=examples/mnist

cd $MNIST_DATA_DIR
sh get_mnist.sh 
cd ../../
$MNIST_DIR/create_mnist.sh 
cd ../
cp ./*prototxt caffe/$MNIST_DIR

exit 0
