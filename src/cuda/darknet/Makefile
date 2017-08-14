GPU?=1
CUDNN?=0
OPENCV?=0
DEBUG?=0
LOGS?=0
#ARCH_I?=35

ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 \
	  -gencode=arch=compute_60,code=sm_60 \
      -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62
#--gpu-architecture=compute_$(ARCH_I) --gpu-code=compute_$(ARCH_I)

VPATH=./src/
TARGET=darknet
OBJDIR=./obj/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CC=gcc
NVCC=/usr/local/cuda/bin/nvcc
OPTS= -Ofast
LDFLAGS= -lm -pthread
COMMON=
CFLAGS= -Wall -Wfatal-errors
NVCCFLAGS= -rdc=true -lcudadevrt

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
	NVCCFLAGS+= -Xcompiler "-DGPU"
	LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1)
	COMMON+= -DCUDNN
	CFLAGS+= -DCUDNN
	LDFLAGS+= -lcudnn
endif

ifeq ($(LOGS), 1)
	CFLAGS+=-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -DLOGS=1 
	#LOG_LINK= -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1 " -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1
	#$(LOGHELPER_LIB)log_helper.o 
#	CFLAGS+= -DLOGS -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda" -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o \
	 im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o \
	 matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o \
	 captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o \
	 yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o \
	 rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o \
	 region_layer.o reorg_layer.o super.o voxel.o args.o log_processing.o
	 
ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o \
	col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o \
	softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o abft.o
endif


NVCCFLAGS+= -Xcompiler "-L/usr/local/cuda/lib64  -lcurand  -lcudart" -Xcompiler \
			'-fPIC' -Xcompiler '-lm' -Wno-deprecated-gpu-targets -Xnvlink --disable-warnings

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

RAD_DIR=/home/carol/radiation-benchmarks

all: obj results $(TARGET)

$(TARGET): $(OBJS) 
#	mv $(OBJDIR)abft.o $(OBJDIR)temp_abft.o $(OBJDIR)temp_abft.o
#	$(NVCC) $(ARCH) -c -dlink -o $(OBJDIR)abft.o  $(OBJDIR)abft.o $(NVCCFLAGS) 
	$(NVCC) -Xcompiler "$(COMMON)"   -lcudadevrt -lcudart $^  -o $@  -Xcompiler "$(LDFLAGS) $(CFLAGS)" $(NVCCFLAGS) \
						-L/usr/local/cuda/lib64  -lcurand -lcudart -lcublas 
	
$(OBJDIR)%.o: %.c $(DEPS) 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(CFLAGS)


$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) -c $< -o $@ $(NVCCFLAGS)

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

test:
	./$(TARGET) -e yolo -m valid -c $(RAD_DIR)/data/darknet/cfg/yolo_v1.cfg \
							-w $(RAD_DIR)/data/darknet/yolo_v1.weights \
							-n 1 -d $(RAD_DIR)/data/gold_caltech.test  \
							-l $(RAD_DIR)/data/networks_img_list/caltech.pedestrians.1K.txt \
							-b $(RAD_DIR)/src/cuda/darknet \
							-x 0 -s 1 -a 0

generate:
	./$(TARGET) -e yolo -m valid -c $(RAD_DIR)/data/darknet/cfg/yolo_v1.cfg \
			-w $(RAD_DIR)/data/darknet/yolo.weights \
			-n 1 -g $(RAD_DIR)/data/gold_caltech.test \
			-l $(RAD_DIR)/data/networks_img_list/caltech.pedestrians.1K.txt \
			-b $(RAD_DIR)/src/cuda/darknet -x 0 -s 1 -a 0


clean:
	rm -rf $(OBJS) $(TARGET) obj/*.o

