# make static lib

all: log_helper.c
	gcc -c log_helper.c
	ar cvr libLogHelper.a log_helper.o

cuda: log_helper.cu
	nvcc -c log_helper.cu
	ar cvr libLogHelperCuda.a log_helper.o

cuda_remote: log_helper_sockclient.cu
	nvcc -c log_helper_sockclient.cu
	ar cvr libLogHelperCuda.a log_helper_sockclient.o
