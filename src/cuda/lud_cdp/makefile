LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: cdplud

cdplud: cdp_lu.cu cdp_lu_main.cu dgetf2.cu dgetrf.cu dlaswp.cu
	nvcc cdp_lu.cu cdp_lu_main.cu dgetf2.cu dgetrf.cu dlaswp.cu $(NVCCFLAGS) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -Xcompiler -fopenmp -lcublas -lcublas_device -lcudadevrt -arch=sm_35 -rdc=true -o cdplud

clean:
	rm -rf cdplud
