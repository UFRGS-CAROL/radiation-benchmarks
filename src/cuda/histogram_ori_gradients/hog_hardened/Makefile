NVIDIA_INC=/usr/local/cuda-7.0/include/
NVIDIA_LIB=/usr/local/cuda-7.0/lib64/
LOGHELPER_INC=../../../include/
LOGHELPER_LIB=../../../include/

NVCC=/usr/local/cuda-7.0/bin/nvcc

ifeq ($(LOGS), 1)
OPTS = -DLOGS
LOG_FLAGS = -I$(LOGHELPER_LIB) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS
endif

debug = -g -G
ALL_CV_FLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_objdetect 
			 #-lopencv_gpu -lopencv_calib3d -lopencv_video -lopencv_legacy   \
			 #-lopencv_stitching -lopencv_superres 
			 #-lopencv_photo  -lopencv_ml -lopencv_ml 			 -lopencv_contrib -lopencv_features2d -lopencv_flann
			#-lopencv_nonfree
ALL_GOLD_GEN=-lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_objdetect  -lopencv_gpu -lopencv_calib3d -lopencv_video -lopencv_legacy   \
			-lopencv_stitching -lopencv_superres \
			-lopencv_photo  -lopencv_ml -lopencv_ml \
			-lopencv_contrib -lopencv_features2d -lopencv_flann \
			-lopencv_nonfree
all:  hog  #gold_generator

hog: HOG.cpp
	$(NVCC) opencv_src/hog.cu -c -I$(NVIDIA_INC)
	$(NVCC) opencv_src/HogDescriptor.cpp -c $(ALL_CV_FLAGS)
	$(NVCC) HOG.cpp -c $(ALL_CV_FLAGS) $(LOG_FLAGS)
	$(NVCC) -o hog_hardened_eccon hog.o HogDescriptor.o HOG.o $(ALL_CV_FLAGS) $(LOG_FLAGS)

gold_generator: GoldGenerator.cpp
	$(NVCC) GoldGenerator.cpp $(debug) -o generator $(ALL_GOLD_GEN)

test:
	./hog_test ../../../../data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data GOLD_1x.data --iterations 1

generate:

	./generator ../../../../data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data GOLD_1x.data --hit_threshold 0.9 --gr_threshold 1 --nlevels 100
	

clean:
	rm -rf hog generator *~ *.data *.jpg ./output/* hog_test *.o
