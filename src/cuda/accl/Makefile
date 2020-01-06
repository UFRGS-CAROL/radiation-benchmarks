CPP = g++

NVCC = /usr/local/cuda/bin/nvcc

SOURCEDIR = .


CPP_SOURCES = $(SOURCEDIR)/accl_main.cpp
CU_SOURCES = $(SOURCEDIR)/accl.cu

LOGHELPER_DIR = ../../include

CFLAGS =

NVCCFLAGS = 

ARCH = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_70,code=sm_70

EXE = $(SOURCEDIR)/accl

DEVICE_OBJS = $(SOURCEDIR)/accl_cu.o

LINK_OBJS = $(SOURCEDIR)/accl_cu_dlink.o

all: accl accl_generate

accl: accl_cu_dlink.o $(CPP_SOURCES)
	$(CPP) -o $(EXE) $(logs) $(DEVICE_OBJS) $(LINK_OBJS) $(CPP_SOURCES) -I$(LOGHELPER_DIR) -L$(LOGHELPER_DIR) -lLogHelper -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lpthread -fopenmp

accl_generate: accl_cu_dlink.o accl_generate.cpp
	$(CPP) -o accl_generate $(DEVICE_OBJS) $(LINK_OBJS) accl_generate.cpp -I$(LOGHELPER_DIR) -L$(LOGHELPER_DIR) -lLogHelper -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lpthread -fopenmp

accl_cu_dlink.o: accl_cu.o
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dlink $(DEVICE_OBJS) -o $(LINK_OBJS) -lcudadevrt

accl_cu.o: $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dc -m64 $(CU_SOURCES) -I$(LOGHELPER_DIR) -o $(DEVICE_OBJS)


generate:
	./accl_generate 

test:
	./(EXE)

clean:
	rm -f $(DEVICE_OBJS)
	rm -f $(LINK_OBJS)
	rm -f $(EXE) accl_generate
