all:
	gcc -fopenmp cache.c ../../include/log_helper.c  -o cache
	gcc -DALL_DEBUG=1 -fopenmp cache.c ../../include/log_helper.c  -o cache_debug

clean:
	rm -r cache cache_debug

