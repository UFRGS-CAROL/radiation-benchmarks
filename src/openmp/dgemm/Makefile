all: dgemm.c
	gcc -DDEFAULTBLOCK=32 -DMKL=0 -DBOFFSET=12 -DVERBOSE=0 -DMAXTHREADS=512 -fopenmp -O3 dgemm.c -o dgemm
