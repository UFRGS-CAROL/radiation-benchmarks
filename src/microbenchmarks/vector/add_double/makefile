all:
	gcc -fopenmp vadd_double.c ../../../include//log_helper.c  -o vadd_double
	gcc -DALL_DEBUG=1 -fopenmp vadd_double.c ../../../include/log_helper.c  -o vadd_double_debug

clean:
	rm -f vadd_double vadd_double_debug

