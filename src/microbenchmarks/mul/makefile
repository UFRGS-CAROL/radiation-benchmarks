all:
	gcc -fopenmp mul.c ../../include/log_helper.c  -o mul
	gcc -DALL_DEBUG=1 -fopenmp mul.c ../../include/log_helper.c  -o mul_debug
	gcc -fopenmp -O0 mul_highlevel.c ../../include/log_helper.c  -o mul_highlevel
	gcc -DALL_DEBUG=1 -O0 -fopenmp mul_highlevel.c ../../include/log_helper.c  -o mul_highlevel_debug

clean:
	rm -f mul mul_debug mul_highlevel mul_highlevel_debug

