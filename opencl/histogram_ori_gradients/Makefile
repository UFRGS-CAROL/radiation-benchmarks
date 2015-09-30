CPP = Main
OPENCV_LIBS= `pkg-config --libs opencv`
ifeq ($(LOGS), 1)
OPTS = -DLOGS
endif

all: clean hog

hog: $(CPP).cpp
	g++ $(OPTS) $(CPP).cpp -Wall -o3  $(OPENCV_LIBS) -o hog

gold_generator: GoldGenerator.cpp
	g++ GoldGenerator.cpp -Wall  -o3 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -l$

clean:
	rm -rf hog gold_gen *~
