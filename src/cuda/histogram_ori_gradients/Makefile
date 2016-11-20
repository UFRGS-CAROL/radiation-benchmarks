NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/
LOGS=1
ifeq ($(LOGS), 1)
OPTS = -DLOGS
LOG_FLAGS =  $(LOGHELPER_INC)log_helper.c -I$(LOGHELPER_LIB)
endif

all: clean hog gold_generator

hog: HOG.cpp
	g++ $(OPTS) HOG.cpp $(LOG_FLAGS) -Wall -o3 -DCXX  -lopencv_core -lopencv_imgproc -lopencv_highgui \
		-lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy \
		-lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres \
		-lopencv_video -lopencv_videostab -o hog

gold_generator: GoldGenerator.cpp
	g++ GoldGenerator.cpp -Wall  -o3 -lopencv_core -I$(LOGHELPER_INC) -DCXX -lopencv_imgproc -lopencv_highgui \
	-lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml \
	-lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video \
	-lopencv_videostab -o gold_gen
	
clean:
	rm -rf hog gold_gen *~ *.data *.jpg ./output/*
