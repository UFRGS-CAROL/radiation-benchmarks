CXX = g++

ALL: sort_gen sort 

sort_gen: sort_generate.cpp
	g++ sort_generate.cpp -lOpenCL -o sort_gen

sort: sort.cpp
	g++ sort.cpp -lOpenCL -o sort




