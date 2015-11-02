# make static lib

all: log_helper.c
	rm -f log_helper.o
	gcc -c log_helper.c
	ar cvr libLogHelper.a log_helper.o

cuda: log_helper.cu
	rm -f log_helper.o
	nvcc -c log_helper.cu
	ar cvr libLogHelperCuda.a log_helper.o

client: log_helper.cu
	rm -f log_helper.o
	rm -f log_helper.cu
	rm -f log_helper_local.cu
	ln -s log_helper_sockclient.c log_helper.cu
	ln -s lhl/log_helper_local.c log_helper_local.cu 
	nvcc -DK1=1 -c log_helper.cu log_helper_local.cu 
	ar cvr libLogHelperCuda.a log_helper.o log_helper_local.o

server: log_helper_sockserver.c
	gcc log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf libLogHelper.a libLogHelperCuda.a *.o
