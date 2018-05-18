#!/bin/bash

set -x

cd caffe/data/mnist/
sh get_mnist.sh 
cd ../../
examples/mnist/create_mnist.sh 

exit 0
