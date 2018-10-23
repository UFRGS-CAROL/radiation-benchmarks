GPU=1
CUDNN=0
OPENCV?=0
OPENMP=0
DEBUG?=0
REAL_TYPE?=float
LOGS?=0
OPENGEMM?=0

RADIATIONDIR=/home/carol/radiation-benchmarks/

ARCH= -gencode arch=compute_60,code=sm_60 \
      -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_62,code=[sm_62,compute_62] \
      -gencode arch=compute_70,code=[sm_70,compute_70] 

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet_v3_$(REAL_TYPE)
OBJDIR=./obj/

CC=gcc
CXX=g++
NVCC=/usr/local/cuda/bin/nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread -lcublas 
LDFLAGS +=  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-write-strings -Werror -fPIC 

NVCCFLAGS=  --std=c++11 
#-disable-warnings 

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
NVCCFLAGS+= -G
endif

CFLAGS+=$(OPTS) -lstdc++
CXXFLAGS+= $(CFLAGS) --std=c++11

ifeq ($(REAL_TYPE), double)
COMMON+= -DREAL_TYPE=64
else
	ifeq ($(REAL_TYPE), half)
	COMMON+= -DREAL_TYPE=16
	else
	COMMON+= -DREAL_TYPE=32
	endif
endif

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

ifeq ($(LOGS), 1)
COMMON += -DLOGS -I$(RADIATIONDIR)/src/include
COMMON += -L$(RADIATIONDIR)/src/include 
LDFLAGS += -lLogHelper
endif

ifeq ($(OPENGEMM), 1)
COMMON += -DOPENGEMM 
NVCCFLAGS += -DOPENGEMM
endif


OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o \
im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o \
network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o \
box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o \
rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o \
l2norm_layer.o yolo_layer.o iseg_layer.o 

# I removed those ones to save time
# captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o nightmare.o instance-segmenter.o
EXECOBJA= yolo.o detector.o  darknet.o detection_gold.o detection_gold_w.o

ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o \
blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o  type.o \
gemm_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results  $(EXEC)

# $(SLIB) #$(ALIB
$(EXEC): $(OBJS) $(EXECOBJ)  
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON)  $(CFLAGS) -c $< -o $@
	
$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON)  $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/* darknet_v3_*

detector:
	./$(EXEC) detector demo cfg/coco.data cfg/yolov3-spp.cfg data/yolov3-spp.weights data/output.avi

demo:
	./$(EXEC) detector test cfg/coco.data cfg/yolov3-spp.cfg data/yolov3-spp.weights data/dog.jpg

generate:
	./$(EXEC) detector test_radiation cfg/coco.data ../../../data/darknet/yolov3.cfg ../../../data/darknet/yolov3.weights \
			../../../data/networks_img_list/caltech.pedestrians.100.txt \
			-generate 1 -gold ./test.csv -tensor_cores 1

test:
	./$(EXEC) detector test_radiation cfg/coco.data ../../../data/darknet/yolov3.cfg ../../../data/darknet/yolov3.weights \
			../../../data/networks_img_list/caltech.pedestrians.100.txt \
			-generate 0 -gold ./test.csv -iterations 1 -norm_coord 0 -tensor_cores 1 -smx_redundancy 1
