CC=g++
CC_FLAGS= -I./mojo/ -std=c++11 -fopenmp -O3 -DMOJO_OPM -DMOJO_SSE3 -msse4 -mavx

intel:CC=icc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-I./mojo/ -std=c++11 -qopenmp -O3 -DMOJO_OPM 


general: mojo_gen mojo_check mojo_err_inj mojo_timing

intel: mojo_gen mojo_check mojo_err_inj mojo_timing

mojo_gen: mojo.cpp
	$(CC) $(CC_FLAGS) mojo.cpp -o mojo_gen

mojo_check: mojo_check.cpp
	$(CC) $(CC_FLAGS) -DLOGS ../../include/log_helper.cpp mojo_check.cpp -o mojo_check

mojo_err_inj: mojo_check.cpp
	$(CC) $(CC_FLAGS) -DLOGS -DERR_INJ ../../include/log_helper.cpp mojo_check.cpp -o mojo_err_inj

mojo_timing: mojo_check.cpp
	$(CC) $(CC_FLAGS) -DTIMING ../../include/log_helper.cpp mojo_check.cpp -o mojo_timing


clean:
	rm -f mojo_gen mojo_check mojo_err_inj mojo_timing
