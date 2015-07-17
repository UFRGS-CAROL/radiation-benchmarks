all: particles particles_gen

particles: TestBed/Demos/Dem2Demo.cpp ../../include/log_helper.c
	g++ -g -o particles_check TestBed/Demos/Dem2Demo.cpp Common/Math/RadixSortHost.cpp ../../include/log_helper.c -DLOGS=1 -I. -ITestBed -ICommon -lOpenCL

particles_gen: TestBed/Demos/input_gold_gen.cpp
	g++ -o particles_gen TestBed/Demos/input_gold_gen.cpp Common/Math/RadixSortHost.cpp -I. -ITestBed -ICommon -lOpenCL
	
clean:
	rm particles_check particles_gen
