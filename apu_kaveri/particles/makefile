all: particles particles_gen

particles: TestBed/Demos/Dem2Demo.cpp
	g++ -g -o particles_check TestBed/Demos/Dem2Demo.cpp Common/Math/RadixSortHost.cpp -I. -ITestBed -ICommon -lOpenCL

particles_gen: TestBed/Demos/input_gold_gen.cpp
	g++ -o particles_gen TestBed/Demos/input_gold_gen.cpp Common/Math/RadixSortHost.cpp -I. -ITestBed -ICommon -lOpenCL
	
clean:
	rm particles_check particles_gen
