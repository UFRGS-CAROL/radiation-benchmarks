NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CPP = Main
OPENCV_LIBS= `pkg-config --libs opencv`

OPTS = -DLOGS
LOG_FLAGS = $(LOGHELPER_INC)log_helper.c -I$(LOGHELPER_LIB)


all: clean hog gold_generator

hog: $(CPP).cpp
	g++ $(OPTS) $(CPP).cpp $(LOG_FLAGS) -Wall -o3 $(OPENCV_LIBS) -o hog_ocl

gold_generator: GoldGenerator.cpp
	g++ GoldGenerator.cpp -Wall  -o3 $(OPENCV_LIBS) -o gold_gen_hog_ocl

clean:
	rm -rf hog_ocl gold_gen_hog_ocl *~ *.data *.jpg ./output/*
