
ifeq ($(LOGS), 1)
OPTS = -DLOGS
endif

all: clean hog

hog: Main.cpp
	g++ $(OPTS) Main.cpp -Wall -o3   -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o hog

gold_generator: GoldGenerator.cpp
	g++ GoldGenerator.cpp -Wall  -o3 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -o gold_gen
	
clean:
	rm -rf hog gold_gen *~