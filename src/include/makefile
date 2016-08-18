# make static lib

all: log_helper.c
	rm -f log_helper.o
	gcc -c log_helper.c
	ar cvr libLogHelper.a log_helper.o

sorts: helper_sorts.cu
	rm -f helper_sorts.o
	nvcc -c helper_sorts.cu
	ar cvr libHelperSorts.a helper_sorts.o

cuda: log_helper.cu sorts
	rm -f log_helper.o
	rm -f log_helper.cu
	ln -s log_helper.c log_helper.cu
	nvcc -c log_helper.cu
	ar cvr libLogHelperCuda.a log_helper.o

client: log_helper.cu
	rm -f log_helper.o
	rm -f log_helper.cu
	ln -s log_helper_sockclient.c log_helper.cu
	nvcc -c log_helper.cu  
	ar cvr libLogHelperCuda.a log_helper.o

server: log_helper_sockserver.c
	gcc log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf libLogHelper.a libLogHelperCuda.a *.o
