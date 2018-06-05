GPU?=1
CUDNN?=0
OPENCV?=0
DEBUG?=0
LOGS?=0
SAFE_MALLOC?=0

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35]
ARCH+= -gencode arch=compute_50,code=[sm_50,compute_50] 
ARCH+= -gencode arch=compute_52,code=[sm_52,compute_52] 
ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] 
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62]

VPATH=./src/
TARGET=darknet_v1
OBJDIR=./obj/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CC=gcc
CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
OPTS= -Ofast
LDFLAGS= -lm -pthread
COMMON= -I../../include/
CFLAGS= -Wall -Wfatal-errors -lpthread
CXXFLAGS=-std=c++11 -lpthread
NVCCFLAGS= -rdc=true -lcudadevrt --default-stream per-thread -ccbin=/usr/bin

CUDAINCLUDE=/usr/local/cuda/include/

ifeq ($(DEBUG), 1)
	OPTS+= -g
	NVCCFLAGS+= -g -G
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
	LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcuda
endif

ifeq ($(CUDNN), 1)
	COMMON+= -DCUDNN
	CFLAGS+= -DCUDNN
	LDFLAGS+= -lcudnn
endif

ifeq ($(GEN_IMG), 1)
	CFLAGS+= -DGEN_IMG
endif

ifeq ($(SAFE_MALLOC), 1)
	CFLAGS+= -DSAFE_MALLOC
endif

ifeq ($(LOGS), 1)
	CFLAGS+=-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -DLOGS=1 
	#LOG_LINK= -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1 " -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS=1
	#$(LOGHELPER_LIB)log_helper.o 
#	CFLAGS+= -DLOGS -Xcompiler "-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda" -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o 
OBJ+= im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o 
OBJ+= matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o 
OBJ+= captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o 
OBJ+= yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o 
OBJ+= rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o 
OBJ+= region_layer.o reorg_layer.o super.o voxel.o args.o log_processing.o abft_host.o copy_network.o
	 
ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o 
OBJ+= col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o 
OBJ+= softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o abft.o maxpool_layer_hardened_kernels.o
endif


NVCCFLAGS+= -Xcompiler "-L/usr/local/cuda/lib64  -lcurand  -lcudart" -Xcompiler 
NVCCFLAGS+=	'-fPIC' -Xcompiler '-lm' -Wno-deprecated-gpu-targets -Xnvlink --disable-warnings

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

RAD_DIR=/home/carol/radiation-benchmarks

all: obj results $(TARGET)

$(TARGET): $(OBJS) 
	$(NVCC) -Xcompiler "$(COMMON)"   -lcudadevrt -lcudart $^  -o $@  -Xcompiler "$(LDFLAGS) $(CFLAGS)" $(NVCCFLAGS) \
						-L/usr/local/cuda/lib64  -lcurand -lcudart -lcublas -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -DLOGS=1 
	
$(OBJDIR)%.o: %.c $(DEPS) 
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(CFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS) 
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(CFLAGS) -I../../include/ $(CXXFLAGS)

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) -c $< -o $@ $(NVCCFLAGS)

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

test_hardening:
	./$(TARGET) test_radiation -e yolo_mem_check -m valid -c $(RAD_DIR)/data/darknet/yolo_v1.cfg \
							-w $(RAD_DIR)/data/darknet/yolo_v1.weights \
							-n 20 -d $(RAD_DIR)/data/darknet/darknet_v1_fault_injection_darknetv1.csv  \
							-l $(RAD_DIR)/data/networks_img_list/fault_injection.txt \
							-b $(RAD_DIR)/src/cuda/darknet_v1 \
							-x 0 -s 0 -a 6
test:
	./$(TARGET) test_radiation -e yolo -m valid -c $(RAD_DIR)/data/darknet/yolo_v1.cfg \
                                                        -w $(RAD_DIR)/data/darknet/yolo_v1.weights \
                                                        -n 20 -d $(RAD_DIR)/data/darknet/darknet_v1_fault_injection_darknetv1.csv  \
                                                        -l $(RAD_DIR)/data/networks_img_list/fault_injection.txt \
                                                        -b $(RAD_DIR)/src/cuda/darknet_v1 \
                                                        -x 1 -s 0 -a 0 -t 1



generate:
	./$(TARGET) test_radiation -e yolo -m valid -c $(RAD_DIR)/data/darknet/yolo_v1.cfg \
			-w $(RAD_DIR)/data/darknet/yolo_v1.weights \
			-n 1 -g $(RAD_DIR)/data/darknet/darknet_v1_fault_injection_darknetv1.csv \
			-l $(RAD_DIR)/data/networks_img_list/fault_injection.txt \
			-b $(RAD_DIR)/src/cuda/darknet_v1 -x 0 -s 1 -a 0 -t 1

test_pure_yolo:
	./$(TARGET) yolo test $(RAD_DIR)/data/darknet/yolo_v1.cfg  $(RAD_DIR)/data/darknet/yolo_v1.weights data/dog.jpg


clean:
	rm -rf $(OBJS) $(TARGET) obj/*.o

