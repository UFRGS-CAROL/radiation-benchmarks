#!/usr/bin/env bash
# raw Makefile configuration

LINE () {
  echo "$@" >> Makefile.config
}

cp Makefile.config.example Makefile.config

#Travis doesn't have CUDA driver installed
LINE ""
LINE "NO_NVML := 1"

LINE "BLAS := open"
LINE "WITH_PYTHON_LAYER := 1"

if $WITH_PYTHON3 ; then
  # TODO(lukeyeager) this path is currently disabled because of test errors like:
  #   ImportError: dynamic module does not define init function (PyInit__caffe)
  LINE "PYTHON_LIBRARIES := python3.4m boost_python-py34"
  LINE "PYTHON_INCLUDE := /usr/include/python3.4 /usr/lib/python3/dist-packages/numpy/core/include"
  LINE "INCLUDE_DIRS := \$(INCLUDE_DIRS) \$(PYTHON_INCLUDE)"
fi

if ! $WITH_IO ; then
  LINE "USE_LEVELDB := 0"
  LINE "USE_LMDB := 0"
fi

if $WITH_CUDA ; then
  # Only build SM50
  LINE "CUDA_ARCH := -gencode arch=compute_50,code=sm_50"
fi

if $WITH_CUDNN ; then
  LINE "USE_CUDNN := 1"
fi

