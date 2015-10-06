NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

CPP = Main
OPENCV_LIBS= `pkg-config --libs opencv`
ifeq ($(LOGS), 1)
OPTS = -DLOGS
endif

all: clean hog gold_generator

hog: $(CPP).cpp
	g++ $(OPTS) $(CPP).cpp $(LOGHELPER_INC)log_helper.c -I$(LOGHELPER_LIB) -Wall -o3 $(OPENCV_LIBS) -o hog

gold_generator: GoldGenerator.cpp
	g++ GoldGenerator.cpp -Wall  -o3 $(OPENCV_LIBS) -o gold_gen

clean:
	rm -rf hog gold_gen *~ *data*
