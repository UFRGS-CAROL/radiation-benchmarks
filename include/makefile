# make static lib

all: log_helper.c
	rm -f log_helper.o
	gcc -c log_helper.c
	ar cvr libLogHelper.a log_helper.o

cuda: log_helper.cu
	rm -f log_helper.o
	nvcc -c log_helper.cu
	ar cvr libLogHelperCuda.a log_helper.o

cuda_remote: log_helper_sockclient.cu
	nvcc -c log_helper_sockclient.cu
	ar cvr libLogHelperCuda.a log_helper_sockclient.o


clean:
	rm -rf libLogHelper.a libLogHelperCuda.a *.o
