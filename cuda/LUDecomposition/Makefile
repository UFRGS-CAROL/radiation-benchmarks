NVCC = nvcc -ccbin g++

# Common includes and paths for CUDA
INCLUDES  = -I./

################################################################################

GENCODE_FLAGS = -gencode arch=compute_50,code=compute_50

NVCC_FLAGS = -dc -m64

LIBRARIES += -lcublas -lcublas_device -lcudadevrt

################################################################################

# Target rules
all: build

build: cdpLUDecomposition

cdp_lu.o:cdp_lu.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

cdp_lu_main.o:cdp_lu_main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

dgetf2.o:dgetf2.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

dgetrf.o:dgetrf.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

dlaswp.o:dlaswp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

cdpLUDecomposition: cdp_lu.o cdp_lu_main.o dgetf2.o dgetrf.o dlaswp.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./cdpLUDecomposition

clean:
	rm -f cdpLUDecomposition cdp_lu.o cdp_lu_main.o dgetf2.o dgetrf.o dlaswp.o

clobber: clean
