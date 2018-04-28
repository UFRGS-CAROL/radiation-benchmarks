#!/usr/bin/env bash
# install dependencies
# (this script must be run as root)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

apt-get -y update
apt-get install -y --no-install-recommends \
  build-essential \
  graphviz \
  libboost-filesystem-dev \
  libboost-python-dev \
  libboost-system-dev \
  libboost-thread-dev \
  libboost-regex-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libhdf5-serial-dev \
  libopenblas-dev \
  libturbojpeg \
  python-virtualenv \
  wget

# package bug WAR:
ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so

if $WITH_CMAKE ; then
  apt-get install -y --no-install-recommends cmake
fi

if ! $WITH_PYTHON3 ; then
  # Python2
  apt-get install -y --no-install-recommends \
    libprotobuf-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-protobuf \
    python-pydot \
    python-skimage
else
  # Python3
  apt-get install -y --no-install-recommends \
    python3-dev \
    python3-numpy \
    python3-skimage \
    python3-pil

  # build Protobuf3 since it's needed for Python3
  echo "Building protobuf3 from source ..."
  pushd .
  PROTOBUF3_DIR=~/protobuf3-build
  rm -rf $PROTOBUF3_DIR
  mkdir $PROTOBUF3_DIR

  # install some more dependencies required to build protobuf3
  apt-get install -y --no-install-recommends \
    curl \
    dh-autoreconf \
    unzip

  wget https://github.com/google/protobuf/archive/3.0.x.tar.gz -O protobuf3.tar.gz
  tar -xzf protobuf3.tar.gz -C $PROTOBUF3_DIR --strip 1
  rm protobuf3.tar.gz
  cd $PROTOBUF3_DIR
  ./autogen.sh
  ./configure --prefix=/usr
  make --jobs=$NUM_THREADS
  make install
  popd
fi

apt-get install -y --no-install-recommends libopencv-dev

if $WITH_IO ; then
  apt-get install -y --no-install-recommends \
    libleveldb-dev \
    liblmdb-dev \
    libsnappy-dev
fi

if $WITH_CUDA ; then
  # install repo packages
  CUDA_REPO_PKG=cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG

  if $WITH_CUDNN ; then
    ML_REPO_PKG=libcudnn7_7.0.5.15-1+cuda8.0_amd64.deb
    ML_REPO_PKGD=libcudnn7-dev_7.0.5.15-1+cuda8.0_amd64.deb
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKGD
    dpkg -i $ML_REPO_PKG $ML_REPO_PKGD
  fi

  # update package lists
  apt-get -y update

  # install packages
  CUDA_PKG_VERSION="8-0"
  CUDA_VERSION="8.0"
  apt-get install -y --no-install-recommends \
    cuda-core-$CUDA_PKG_VERSION \
    cuda-cudart-dev-$CUDA_PKG_VERSION \
    cuda-cublas-dev-$CUDA_PKG_VERSION \
    cuda-nvml-dev-$CUDA_PKG_VERSION   \
    cuda-curand-dev-$CUDA_PKG_VERSION
  # manually create CUDA symlink
  ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

  if $WITH_CUDNN ; then
    apt-get install -y --no-install-recommends libcudnn7 libcudnn7-dev
  fi
fi

