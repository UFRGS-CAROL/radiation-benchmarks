CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 
EXEC=cudaACCL
LOGS=1
BUILDPROFILER=0
ITERATIONS=10
SIZE=7
FRAMES=7
GOLD=./gold_$(SIZE)_$(FRAMES).data
#../gold_accl/gold_7_7 
INPUT=../../../data/accl/7Frames.pgm

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -Xptxas -v -dc  -Wno-deprecated-gpu-targets

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler

# check for cuda 7.0 or lower
ifneq (, $(findstring cuda-7.0, $(CUDAPATH)))
else
# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]
# Tegra X2
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]
# Titan V | Tesla V100
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]
# Xavier
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]
endif

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

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
LDFLAGS+= -L../../include  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L../common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif


NVCCFLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(patsubst %-Xptxas,%,$(NVCCOPTFLAGS)))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

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
	$(NVCC) -ccbin $(CXX) $(ARCH)  -dlink $@ -o $@_dlink.o -lcudadevrt

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	

generate:
	./$(EXEC) --size $(SIZE) --frames $(FRAMES) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose --generate

test:
	./$(EXEC) --size $(SIZE) --frames $(FRAMES) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose
	
