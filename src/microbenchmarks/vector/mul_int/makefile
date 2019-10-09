all:
	gcc -fopenmp vmul_int.c ../../../include//log_helper.c  -o vmul_int
	gcc -DALL_DEBUG=1 -fopenmp vmul_int.c ../../../include/log_helper.c  -o vmul_int_debug

clean:
	rm -f vmul_int vmul_int_debug

