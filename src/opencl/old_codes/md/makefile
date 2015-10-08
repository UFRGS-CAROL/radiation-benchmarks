all: md_check gen

md_check: md_check.cpp md_ocl.cpp md.h
	g++ md_check.cpp md_ocl.cpp -lOpenCL -fopenmp -o md_check

gen: md_gen.cpp md_ocl.cpp md.h
	g++ md_gen.cpp md_ocl.cpp -lOpenCL -fopenmp -o md_gen
