all: cdpLUDecomposition

cdpLUDecomposition: cdp_lu.cu cdp_lu_main.cu dgetf2.cu dgetrf.cu dlaswp.cu
	nvcc cdp_lu.cu cdp_lu_main.cu dgetf2.cu dgetrf.cu dlaswp.cu -o cdpLUDecomposition -Xcompiler -fopenmp -lcublas -lcublas_device -lcudadevrt -arch=sm_35 -rdc=true -I.

clean:
	rm -rf cdpLUDecomposition
