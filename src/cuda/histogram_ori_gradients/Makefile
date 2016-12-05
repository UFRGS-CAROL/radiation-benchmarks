NVCC=/usr/local/cuda/bin/nvcc
CXX=g++

ARCH_I?=35

ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 \
      -gencode arch=compute_53,code=sm_53 -gencode=arch=compute_60,code=sm_60 \
      -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62
# --gpu-architecture=compute_$(ARCH_I) --gpu-code=compute_$(ARCH_I)

ifeq ($(DBG), 1)
	DEBUG = -g -G
endif

ifeq ($(LOGS), 1)
	OPTS = -DLOGS
	LOG_FLAGS =  -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda
endif


NVIDIA_INC=/usr/local/cuda/include/ $(ARCH)
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CFLAGS+= -L$(NVIDIA_LIB) -lcudart


NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

#dirs
HOG_EXT_DIR = ./hog_extracted
HOG_EVA_DIR = ./hog_evaluate_dataset
HOG_HAR_DIR = ./hog_hardened_ecc_on
HOG_EOF_DIR = ./hog_hardened_ecc_off
HOG_OCV_DIR = ./hog_opencv



ALL_CV_FLAGS=-L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_objdetect

ALL_GOLD_GEN=`pkg-config opencv --cflags --libs` 

ALL_CV_FLAGS += $(OPTS) 

.PHONY: all clean gold_generator hog_extracted hog_har_ecc_off hog_har_ecc_on hog_opencv hog_eval generator_ext
all: clean gold_generator hog_extracted hog_har_ecc_off hog_har_ecc_on hog_opencv hog_eval generator_ext


##Opencv HOG
hog_opencv: $(HOG_OCV_DIR)/HOG.cpp
	$(CXX) $(OPTS) $(HOG_OCV_DIR)/HOG.cpp $(LOG_FLAGS) -Wall -DCXX -o3 $(ALL_GOLD_GEN) -o $(HOG_OCV_DIR)/hog_opencv -I$(LOGHELPER_INC)\
		 -L/usr/local/cuda/lib64/  -lcudart 


##hog extracted
hog_extracted: $(HOG_EXT_DIR)/HOG.cpp
	$(NVCC) $(HOG_EXT_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_EXT_DIR)/hog.o
	$(CXX) $(HOG_EXT_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_EXT_DIR)/HogDescriptor.o $(LOG_FLAGS) $(CFLAGS)
	$(CXX) $(HOG_EXT_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) -o $(HOG_EXT_DIR)/HOG.o $(LOG_FLAGS) $(CFLAGS)
	$(CXX) -o $(HOG_EXT_DIR)/hog_ext $(HOG_EXT_DIR)/hog.o $(HOG_EXT_DIR)/HOG.o $(HOG_EXT_DIR)/HogDescriptor.o \
							$(ALL_CV_FLAGS) $(CFLAGS) $(LOG_FLAGS)


##hog hardened on
hog_har_ecc_on: $(HOG_HAR_DIR)/HOG.cpp
	$(NVCC) $(HOG_HAR_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_HAR_DIR)/hog.o
	$(NVCC) $(HOG_HAR_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_HAR_DIR)/HogDescriptor.o $(LOG_FLAGS) $(CFLAGS)
	$(NVCC) $(HOG_HAR_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS) -o $(HOG_HAR_DIR)/HOG.o $(LOG_FLAGS) $(CFLAGS)
	$(CXX) -o $(HOG_HAR_DIR)/hog_har_eccon $(HOG_HAR_DIR)/hog.o $(HOG_HAR_DIR)/HogDescriptor.o $(HOG_HAR_DIR)/HOG.o  \
							$(ALL_CV_FLAGS) $(CFLAGS) $(LOG_FLAGS)


##hog hardened ecc off
hog_har_ecc_off: $(HOG_EOF_DIR)/HOG.cpp
	$(NVCC) $(HOG_EOF_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_EOF_DIR)/hog.o
	$(NVCC) $(HOG_EOF_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_EOF_DIR)/HogDescriptor.o $(LOG_FLAGS) $(CFLAGS)
	$(NVCC) $(HOG_EOF_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS) -o $(HOG_EOF_DIR)/HOG.o $(LOG_FLAGS) $(CFLAGS)
	$(CXX) -o $(HOG_EOF_DIR)/hog_har_eccoff $(HOG_EOF_DIR)/hog.o $(HOG_EOF_DIR)/HogDescriptor.o $(HOG_EOF_DIR)/HOG.o  \
							$(ALL_CV_FLAGS) $(CFLAGS) $(LOG_FLAGS)


##hog evaluate
hog_eval: $(HOG_EVA_DIR)/GoldGenerator.cpp
	$(CXX) $(HOG_EVA_DIR)/GoldGenerator.cpp -Wall  -Ofast -lopencv_core -I$(LOGHELPER_INC) $(ALL_GOLD_GEN) -o $(HOG_EVA_DIR)/hog_evaluate_dataset


gold_generator: $(HOG_OCV_DIR)/GoldGenerator.cpp
	g++ $(HOG_OCV_DIR)/GoldGenerator.cpp -Wall  -o3 -lopencv_core -I$(LOGHELPER_INC) $(ALL_GOLD_GEN) -o $(HOG_OCV_DIR)/gold_gen	

generator_ext:  $(HOG_EXT_DIR)/GoldGenerator.cpp
	g++ $(HOG_EXT_DIR)/GoldGenerator.cpp -Wall  -o3 -lopencv_core -I$(LOGHELPER_INC)  $(ALL_GOLD_GEN) -o $(HOG_EXT_DIR)/gold_gen
	
test:
	.$(HOG_EVA_DIR)/hog_evaluate_dataset ./dataset_example.txt --dst_data dataset.txt --hit_threshold 0.9 --gr_threshold 1 --nlevels 100

generate:
	./generator dataset_example.txt --hit_threshold 0.9 --gr_threshold 1 --nlevels 100
	
	
clean:
	rm -rf  $(HOG_EXT_DIR)/hog_ext $(HOG_OCV_DIR)/hog_opencv $(HOG_HAR_DIR)/hog_har_eccon  $(HOG_EOF_DIR)/hog_har_eccoff $(HOG_OCV_DIR)/gold_gen $(HOG_EXT_DIR)/gold_gen \
			$(HOG_EXT_DIR)/*.o $(HOG_OCV_DIR)/*.o $(HOG_HAR_DIR)/*.o  $(HOG_EOF_DIR)/*.o $(HOG_OCV_DIR)/*.o
