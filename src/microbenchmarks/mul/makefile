all:
	gcc -fopenmp mul.c ../../include/log_helper.c  -o mul
	gcc -DALL_DEBUG=1 -fopenmp mul.c ../../include/log_helper.c  -o mul_debug

clean:
	rm -f mul mul_debug

