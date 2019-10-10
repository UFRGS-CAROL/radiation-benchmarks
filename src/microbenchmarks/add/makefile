all:
	gcc -fopenmp add.c ../../include/log_helper.c  -o add
	gcc -DALL_DEBUG=1 -fopenmp add.c ../../include/log_helper.c  -o add_debug
	gcc -fopenmp add_highlevel.c ../../include/log_helper.c  -o add_highlevel
	gcc -DALL_DEBUG=1 -fopenmp add_highlevel.c ../../include/log_helper.c  -o add_debug_highlevel

clean:
	rm -f add add_debug add_highlevel add_debug_highlevel

