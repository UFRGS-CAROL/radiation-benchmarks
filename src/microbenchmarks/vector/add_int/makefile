all:
	gcc -fopenmp vadd_int.c ../../../include//log_helper.c  -o vadd_int
	gcc -DALL_DEBUG=1 -fopenmp vadd_int.c ../../../include/log_helper.c  -o vadd_int_debug

clean:
	rm -f vadd_int vadd_int_debug

