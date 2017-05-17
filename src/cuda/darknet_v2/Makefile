GPU?=1
CUDNN=0
OPENCV=0
DEBUG=0

#for radiation setup
LOGS?=0
GEN_IMG?=0 #if you want to generate images on gold generator execution

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CXX=g++
NVCC=nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors

ifeq ($(GEN_IMG), 1)
CFLAGS+= -DGEN_IMG=1
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o \
list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o \
dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o \
connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o \
captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o \
coco.o dice.o yolo.o detector.o layer.o compare.o regressor.o classifier.o local_layer.o \
swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o \
crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o \
lsd.o super.o voxel.o tree.o args.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o \
im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o \
maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o maxpool_layer_hardened_kernels.o
endif

ifeq ($(LOGS), 1)
LOGHELPER_INC=-I../../include/
LOGHELPER_LIB= -L../../include/ -lLogHelper -DLOGS=1
HELPFUL=-I../../include/
else
HELPFUL=-I../../include/
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ)) $(OBJDIR)log_processing.o
DEPS = $(wildcard src/*.h) Makefile

all: obj backup results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ $(LOGHELPER_INC) $(LOGHELPER_LIB) -o $@  $(LDFLAGS) -lstdc++  

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@  $(HELPFUL)

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@ 

$(OBJDIR)log_processing.o: src/log_processing.cpp src/log_processing.h
	$(CXX)  src/log_processing.cpp -c -o obj/log_processing.o $(LOGHELPER_LIB) $(LOGHELPER_INC) $(HELPFUL) $(OPTS) $(CFLAGS) $(COMMON) $(LDFLAGS)

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)


test_pure_yolo:
	./darknet detect cfg/yolo.cfg ../../../data/darknet/yolo_v2.weights data/dog.jpg
	
test:
	./darknet test_radiation -d temp.csv -n 2 -s 1

generate:
	./darknet test_radiation  -c cfg/yolo.cfg -w ../../../data/darknet/yolo_v2.weights -g 1 -d temp.csv -s 1 -l test.txt 

