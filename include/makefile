# make static lib

all: log_helper.c
	rm -f log_helper.o
	gcc -c log_helper.c
	ar cvr libLogHelper.a log_helper.o

cuda: log_helper.cu
	rm -f log_helper.o
	nvcc -c log_helper.cu
	ar cvr libLogHelperCuda.a log_helper.o

<<<<<<< HEAD
cuda_remote: log_helper_sockclient.cu
	nvcc -c log_helper_sockclient.cu
	ar cvr libLogHelperCuda.a log_helper_sockclient.o


=======
>>>>>>> 17d8ccaaf5750e2a5f908317006d531d8570e960
clean:
	rm -rf libLogHelper.a libLogHelperCuda.a *.o
