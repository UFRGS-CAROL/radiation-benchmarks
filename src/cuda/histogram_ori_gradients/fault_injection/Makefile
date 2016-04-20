NVIDIA_INC=/usr/local/cuda/include/,/usr/local/include/opencv,/usr/local/include
NVIDIA_LIB=/usr/local/cuda/lib64/,/usr/local/lib,/usr/local/share/OpenCV/3rdparty/lib
LOGHELPER_INC=../../../include/
LOGHELPER_LIB=../../../include/

ifeq ($(LOGS), 1)
OPTS = -DLOGS
LOG_FLAGS =  $(LOGHELPER_INC)log_helper.cu -I$(LOGHELPER_LIB)
endif

debug = -g -G

ALL_COMMANDS=$(OPTS) $(LOG_FLAGS) $(debug) -L$(NVIDIA_LIB) -I$(NVIDIA_INC) -lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_gpu -lopencv_objdetect -lopencv_highgui
ALL_NOD=$(OPTS) $(LOG_FLAGS) -L$(NVIDIA_LIB) -I$(NVIDIA_INC) -lopencv_core -lopencv_imgproc -lopencv_highgui  -lopencv_gpu -lopencv_objdetect -lopencv_highgui

all: hog_without gold_hog_without

gold_hog_without:
	nvcc -arch=sm_35 $(ALL_COMMANDS) -c GoldGenerator.cu
	nvcc -arch=sm_35 $(ALL_COMMANDS) -c hog.cu
	nvcc -arch=sm_35 $(ALL_COMMANDS) -o gold_hog_without GoldGenerator.o hog.o
	rm -rf GoldGenerator.o hog.o

hog_without:
	nvcc -arch=sm_35 $(ALL_COMMANDS) -c HOG.cu
	nvcc -arch=sm_35 $(ALL_COMMANDS) -c hog.cu
	nvcc -arch=sm_35 $(ALL_COMMANDS) -o hog_without HOG.o hog.o
	
clean:
	rm -rf hog_without gold_hog_without *~ *.data *.o output*.jpg
	
