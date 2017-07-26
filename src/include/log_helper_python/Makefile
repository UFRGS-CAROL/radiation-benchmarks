
all:log_helper


	

log_helper: log_helper.c log_helper.i
	swig -python log_helper.i
	gcc  -fPIC -c log_helper.c log_helper_wrap.c -I/usr/include/python2.7/ -I../
	gcc -shared log_helper.o log_helper_wrap.o -o _log_helper.so

clean:
	rm -rf *.o *py *so
