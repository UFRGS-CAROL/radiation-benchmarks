GPU?=0
CUDNN?=0
OPENCV?=0
DEBUG?=0
LOGS?=0
ARCH_I?=35

ARCH= --gpu-architecture=compute_$(ARCH_I) --gpu-code=compute_$(ARCH_I)

VPATH=./src/
EXEC=darknet
OBJDIR=./obj/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CC=gcc
NVCC=/usr/local/cuda/bin/nvcc
OPTS= -Xcompiler "-Ofast"
LDFLAGS= -Xcompiler "-lm -pthread"
COMMON=
CFLAGS= -Xcompiler "-Wall -Wfatal-errors" -Wno-deprecated-gpu-targets

CUDAINCLUDE=/usr/local/cuda-7.0/include/

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
	CFLAGS+= -Xcompiler "-DGPU"
	LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1)
	COMMON+= -DCUDNN
	CFLAGS+= -DCUDNN
	LDFLAGS+= -lcudnn
endif

ifeq ($(LOGS), 1)
	#CFLAGS+=  -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1 " -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1 
	#LOG_LINK= -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1 " -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1
	#$(LOGHELPER_LIB)log_helper.o 
	LOG_CU = $(LOGHELPER_LIB)log_helper.c
#	CFLAGS+= -DLOGS -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda" -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o \
	 im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o \
	 matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o \
	 captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o \
	 yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o \
	 rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o \
	 region_layer.o reorg_layer.o super.o voxel.o log_processing.o args.o
	 
ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o \
	col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o \
	softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

#abft = 1 is dumb abft
#abft = 2 is fast abft
#LOG_LINK+= -Xcompiler "-DABFT=1"
BUILD_ABFT=abft
ABFT_SRC=src/abft.cu
#CFLAGS += -DABFT=1
BUILD_ABFT_OBJ=$(OBJDIR)abft.o
 
#$(NVCC) -I$(CUDAINCLUDE) $(NVCCFLAGS)  -dlink $(OBJDIR)abft_temp.o  -lcudadevrt -o $(OBJDIR)abft.o   $(ARCH)  

NVCCFLAGS= -Xcompiler "-L/usr/local/cuda/lib64 -lcuda -lcurand  -lcudart" -Xcompiler '-fPIC' -Xcompiler '-lm'

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: build_logs obj results $(EXEC)

abft: src/abft.cu
	$(NVCC)  -I$(CUDAINCLUDE) $(NVCCFLAGS) -c -O3 -dc src/abft.cu -o $(OBJDIR)abft.o -rdc=true $(ARCH)  -lcudadevrt 
	

$(EXEC): $(OBJS) 
	$(NVCC) $(COMMON) $(BUILD_ABFT_OBJ) $(OBJDIR)log_helper.o $^  -o $@  -lcudadevrt -lcudart -lcuda -rdc=true  $(LDFLAGS) $(CFLAGS) -DLOGS -I../../include
	
$(OBJDIR)%.o: %.c $(DEPS) 
	$(NVCC) $(COMMON) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(CFLAGS)  -DLOGS  -I../../include

build_logs: $(LOG_CU)
	$(NVCC)  $(LOG_CU) -c -o $(OBJDIR)log_helper.o

$(OBJDIR)%.o: %.cu $(DEPS) $(BUILD_ABFT)  build_logs
	$(NVCC) $(ARCH) $(COMMON) $(CFLAGS) -c $< -o $@  $(LDFLAGS) $(CFLAGS)

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

test:darknet
	./darknet -e yolo -m valid -c cfg/yolo.cfg -w data/yolo.weights -n 3 -d data/gold_voc2012.test \
							-l data/voc.2012.DEBUG.txt -b ~/radiation-benchmarks/src/cuda/darknet -x 0 -s 1 -a 1

generate:darknet
	./darknet -e yolo -m valid -c cfg/yolo.cfg -w data/yolo.weights -n 1 -g data/gold_voc2012.test \
			-l data/voc.2012.DEBUG.txt -b ~/radiation-benchmarks/src/cuda/darknet -x 0 -s 0 -a 0


clean:
	rm -rf $(OBJS) $(EXEC) obj/*.o

