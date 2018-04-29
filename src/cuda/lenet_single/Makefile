CXX=/usr/bin/g++
CUDADIR=/usr/local/cuda-9.1
NVCC=$(CUDADIR)/bin/nvcc
CAFFEDIR=caffe/
TARGET=lenet

ARCH=  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 
ARCH+= -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 
ARCH+= -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70


CXXFLAGS= -MMD -MP -pthread -fPIC -DCAFFE_VERSION=0.17.0 
CXXFLAGS+= -std=c++11 -DCUDA_NO_HALF -DNDEBUG -O3 -DUSE_CUDNN -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB
CXXFLAGS+= -Wall -Wno-sign-compare -pthread -fPIC -Wl,-rpath,\$ORIGIN/../../lib
		 
NVCCFLAGS= -Xptxas -v $(ARCH)

LIBRARY=  -L$(CUDADIR)/lib64   -L/usr/local/lib -L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial 
LIBRARY+=  -L/usr/lib/nvidia-396 -L/usr/lib/nvidia-390 -L/usr/lib/nvidia-387 
LIBRARY+= -L/usr/lib/nvidia-384 -L/usr/lib/nvidia-381 -L/usr/lib/nvidia-375 -L/usr/lib/nvidia-367 
LIBRARY+= -L/usr/lib/nvidia-361 -L/usr/lib/nvidia-352 -L$(CUDADIR)/lib  
LIBRARY+= -L/usr/lib/x86_64-linux-gnu/hdf5/serial -L/usr/lib/openblas-base/ -L$(CAFFEDIR)/.build_release/lib  -L$(CAFFEDIR)/build/lib

INCLUDE=  -I$(CUDADIR)/include -I/usr/local/include -I/usr/include/hdf5/serial -I$(CAFFEDIR)/build/src 
INCLUDE+= -I$(CAFFEDIR)/src -I$(CAFFEDIR)/include -I$(CAFFEDIR)/3rdparty -I/usr/include/hdf5/serial  
INCLUDE+= -I$(CUDADIR)/include -I/usr/include/openblas/


LDFLAGS=  -lcuda -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system 
LDFLAGS+= -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++  
LDFLAGS+= -lcblas -latlas -lnvidia-ml -lboost_system -lgflags -lprotobuf -lboost_filesystem 
LDFLAGS+= -lm -lturbojpeg -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lopencv_core 
LDFLAGS+= -lboost_thread -lboost_regex -lstdc++ -lcudnn -lopenblas 

DATAPATH=/home/carol/radiation-benchmarks/data/lenet
SRCRAD=/home/carol/radiation-benchmarks/src/cuda/lenet_single

ifeq ($(DEBUG), 1)
	NVCCFLAGS+= -G -g
	CXXFLAGS+= -g
endif

ifeq ($(LOGS), 1)
INCLUDE+= -I../../include/
LDFLAGS+= -L../../include/ -lLogHelper
CXXFLAGS+= -DLOGS=1
endif

OBJ=lenet_single.o logs_processing.o


OBJDIR=./obj/
VPATH=./src/

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

BUILDFLAGS = $(LDFLAGS) $(CXXFLAGS) $(INCLUDE) $(LIBRARY) 

.PHONY : all clean
all: obj $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(BUILDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) -c $< -o $@ $(BUILDFLAGS)

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) --compiler-options "$(CXXFLAGS)" -c $< -o $@  $(NVCCFLAGS) $(INCLUDE)

obj:
	mkdir -p obj

clean:
	rm -rf $(TARGET) $(OBJDIR)/*.o $(OBJDIR)/*.d

	
	
test:
	 env LD_LIBRARY_PATH=/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/build/lib/ \
	 ./lenet test -model=$(DATAPATH)/lenet_train_test.prototxt \
	 	-weights=$(DATAPATH)/single_iter_10000.caffemodel \
	 	1000 1 $(DATAPATH)/gold_test.csv
			 	
train:
	 env LD_LIBRARY_PATH=$(SRCRAD)/caffe/build/lib/ \
	 ./lenet train --solver=$(DATAPATH)/lenet_solver_single.prototxt
	rm -rf _iter_10000.caffemodel  _iter_10000.solverstate

	
