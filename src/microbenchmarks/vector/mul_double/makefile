all:
	gcc -fopenmp -lm vmul_double.c ../../../include//log_helper.c  -o vmul_double
	gcc -DALL_DEBUG=1 -fopenmp -lm vmul_double.c ../../../include/log_helper.c  -o vmul_double_debug

clean:
	rm -f vmul_double vmul_double_debug

