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
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON=
CFLAGS=-Wall -Wfatal-errors

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

ifeq ($(LOGS), 1)
	CFLAGS+= -DLOGS -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper
	LOG_LINK= $(LOGHELPER_LIB)log_helper.o 
endif

#abft = 1 is dumb abft
#abft = 2 is fast abft
ifeq ($(ABFT),1)
	CFLAGS+= -DABFT=1
else ifeq($ABFT),2)
	CFLAGS+= -DABFT=2
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


OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LOG_LINK)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

test:darknet
	./darknet -e yolo -m valid -c cfg/yolo.cfg -w yolo.weights -n 3 -d gold/gold_voc2012.test -l /home/carol/radiation-benchmarks/data/VOC2012/voc.2012.ONLY50.txt -b gold/comp4_det_test_ -x 0

generate:darknet
	./darknet -e yolo -m valid -c cfg/yolo.cfg -w yolo.weights -n 1 -g gold/gold_voc2012.test -l /home/carol/radiation-benchmarks/data/VOC2012/voc.2012.ONLY50.txt -b gold/comp4_det_test_ -x 0

clean:
	rm -rf $(OBJS) $(EXEC) obj/*.o

