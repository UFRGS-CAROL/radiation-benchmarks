all:
	gcc -fopenmp add.c ../../include/log_helper.c  -o add
	gcc -DALL_DEBUG=1 -fopenmp add.c ../../include/log_helper.c  -o add_debug

clean:
	rm -f add add_debug

