NVCC=/usr/local/cuda-7.0/bin/nvcc
CXX=g++

ifeq ($(DBG), 1)
	DEBUG = -g -G
endif

ifeq ($(LOGS), 1)
	OPTS = -DLOGS
	LOG_FLAGS =  -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda
endif


NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/



NVIDIA_INC=/usr/local/cuda-7.0/include/
NVIDIA_LIB=/usr/local/cuda-7.0/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

#dirs
HOG_EXT_DIR = ./hog_extracted
HOG_EVA_DIR = ./hog_evaluate_dataset
HOG_HAR_DIR = ./hog_hardened_ecc_on
HOG_EOF_DIR = ./hog_hardened_ecc_off
HOG_OCV_DIR = ./hog_opencv



ALL_CV_FLAGS=-L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_objdetect 

ALL_GOLD_GEN= -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_objdetect  -lopencv_gpu -lopencv_calib3d -lopencv_video -lopencv_legacy \
			-lopencv_stitching -lopencv_superres \
			-lopencv_photo  -lopencv_ml -lopencv_ml \
			-lopencv_contrib -lopencv_features2d -lopencv_flann \
			-lopencv_nonfree

ALL_CV_FLAGS += $(OPTS) 


all: clean gold_generator hog_extracted hog_har_ecc_off hog_har_ecc_on hog_opencv

OPENCV_FLAGS=  -lopencv_core -lopencv_imgproc -lopencv_highgui \
		-lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy \
		-lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres \
		-lopencv_video -lopencv_videostab
		


##Opencv HOG
hog_opencv: $(HOG_OCV_DIR)/HOG.cpp
	$(CXX) $(OPTS) $(HOG_OCV_DIR)/HOG.cpp $(LOG_FLAGS) -Wall -DCXX -o3 $(OPENCV_FLAGS) -o $(HOG_OCV_DIR)/hog_opencv -I$(LOGHELPER_INC)\
		 -L/usr/local/cuda/lib64/ -lcuda -lcudart 


##hog extracted
hog_extracted: $(HOG_EXT_DIR)/HOG.cpp
	$(NVCC) $(HOG_EXT_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_EXT_DIR)/hog.o
	$(NVCC) $(HOG_EXT_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_EXT_DIR)/HogDescriptor.o
	$(NVCC) $(HOG_EXT_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS) -o $(HOG_EXT_DIR)/HOG.o
	$(NVCC) -o $(HOG_EXT_DIR)/hog_ext $(HOG_EXT_DIR)/hog.o $(HOG_EXT_DIR)/HogDescriptor.o $(HOG_EXT_DIR)/HOG.o $(ALL_CV_FLAGS) $(LOG_FLAG) 


##hog hardened on
hog_har_ecc_on: $(HOG_HAR_DIR)/HOG.cpp
	$(NVCC) $(HOG_HAR_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_HAR_DIR)/hog.o
	$(NVCC) $(HOG_HAR_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_HAR_DIR)/HogDescriptor.o
	$(NVCC) $(HOG_HAR_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS) -o $(HOG_HAR_DIR)/HOG.o
	$(NVCC) -o $(HOG_HAR_DIR)/hog_har_eccon $(HOG_HAR_DIR)/hog.o $(HOG_HAR_DIR)/HogDescriptor.o $(HOG_HAR_DIR)/HOG.o $(ALL_CV_FLAGS) $(LOG_FLAGS)


##hog hardened ecc off
hog_har_ecc_off: $(HOG_EOF_DIR)/HOG.cpp
	$(NVCC) $(HOG_EOF_DIR)/opencv_src/hog.cu -c -I$(NVIDIA_INC) -o $(HOG_EOF_DIR)/hog.o
	$(NVCC) $(HOG_EOF_DIR)/opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS) -o $(HOG_EOF_DIR)/HogDescriptor.o
	$(NVCC) $(HOG_EOF_DIR)/HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS) -o $(HOG_EOF_DIR)/HOG.o
	$(CXX) -o $(HOG_EOF_DIR)/hog_har_eccoff $(HOG_EOF_DIR)/hog.o $(HOG_EOF_DIR)/HogDescriptor.o $(HOG_EOF_DIR)/HOG.o \
		$(ALL_CV_FLAGS) $(LOG_FLAGS) -L/usr/local/cuda-7.0/lib64 -lcuda -lcudart -lcublas -lcurand


##hog evaluate
hog_eval: $(HOG_EVA_DIR)/GoldGenerator.cpp
	$(CXX) $(HOG_EVA_DIR)/GoldGenerator.cpp -Wall  -o3 -lopencv_core -I$(LOGHELPER_INC) -lopencv_imgproc -lopencv_highgui \
	-lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml \
	-lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video \
	-lopencv_videostab -o $(HOG_EVA_DIR)/hog_evaluate_dataset




gold_generator: $(HOG_OCV_DIR)/GoldGenerator.cpp
	g++ $(HOG_OCV_DIR)/GoldGenerator.cpp -Wall  -o3 -lopencv_core -I$(LOGHELPER_INC)  -lopencv_imgproc -lopencv_highgui \
	-lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml \
	-lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video \
	-lopencv_videostab -o $(HOG_OCV_DIR)/gold_gen	
	
	
test:
	./hog dataset_example.txt --iterations 1

generate:
	./generator dataset_example.txt --hit_threshold 0.9 --gr_threshold 1 --nlevels 100
	
	
clean:
	rm -rf  $(HOG_EXT_DIR)/hog_ext $(HOG_OCV_DIR)/hog $(HOG_HAR_DIR)/hog_har_eccon  $(HOG_EOF_DIR)/hog_har_eccoff $(HOG_OCV_DIR)/gold_gen \
			$(HOG_EXT_DIR)/*.o $(HOG_OCV_DIR)/*.o $(HOG_HAR_DIR)/*.o  $(HOG_EOF_DIR)/*.o $(HOG_OCV_DIR)/*.o
