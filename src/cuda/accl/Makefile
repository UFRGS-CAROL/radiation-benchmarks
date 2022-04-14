CUDAPATH=/usr/local/cuda
CC=gcc
CXX=g++
NVCC=$(CUDAPATH)/bin/nvcc
LOGS=1
USE_OMP=1

EXEC=cudaACCL
BUILDPROFILER=0
ITERATIONS=10
SIZE=7
FRAMES=7
RADDIR=../../..
DATADIR=$(RADDIR)/data/accl
GOLD=$(DATADIR)/gold_$(SIZE)_$(FRAMES).data
INPUT=$(DATADIR)/7Frames.pgm

VPATH=./src/

CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 
NVCCFLAGS= -std=c++11 -rdc=true

# Kepler, Volta
SMS= 35 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/libLogHelper/include -I$(RADDIR)/src/cuda/common/include

OBJDIR=./obj/
GPUOBJ= accl.o_dlink.o
OBJ= accl.o main.o Parameters.o


ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/libLogHelper/  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt

OBJS = $(addprefix $(OBJDIR), $(OBJ))
GPUOBJS = $(addprefix $(OBJDIR), $(GPUOBJ))
DEPS = $(wildcard src/*.h) Makefile

all: mkdir $(EXEC)

$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $(GPUOBJS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE)
	$(NVCC) -ccbin $(CXX) $(ARCH) -dlink $@ -o $@_dlink.o -lcudadevrt

mkdir:
	mkdir -p $(OBJDIR)
	mkdir -p $(DATADIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	

generate:
	./$(EXEC) --size $(SIZE) --frames $(FRAMES) --input $(INPUT) --gold $(GOLD) --iterations 1 --verbose --generate

test:
	./$(EXEC) --size $(SIZE) --frames $(FRAMES) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose
	
