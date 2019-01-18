#!/bin/bash

#install pip
sudo apt install python-pip -y

# install cython
pip install Cython

cp ../killtestSignal.py .

sed 

rm -rf killtestSignal.c killtestSignal.so build/

sed -i 's/\#!\/usr\/bin\/python/\# distutils\: language = c++/g' killtestSignal.py

cython killtestSignal.py --embed --cplus -o killtestSignal.cpp




