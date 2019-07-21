LOGS=1
DEBUG?=0
PRECISION=double
REDUNDANCY=dmrmixed
CHECKBLOCK=0
ZERO_HALF=2e-3
ZERO_FLOAT=2e-4

RADDIR=/home/carol/radiation-benchmarks


CXX=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]	# Jetson Xavier

NVCCFLAGS= $(ARCH) -Xptxas -v -std=c++11
CXXFLAGS= -std=c++11 -fopenmp -O3
INCLUDE= -I/usr/local/cuda/include/ -I$(RADDIR)/src/include/ -I$(RADDIR)/src/cuda/common/include/
LIBRARY= -L/usr/local/cuda/lib64/  -L$(RADDIR)/src/include/ 

OBJDIR=./obj/

EXEC=cuda_hotspot_mp


ifeq ($(DEBUG), 1)
NVCCFLAGS+= -g -G
CXXFLAGS+= -g
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
LIBRARY+= -lLogHelper
endif

COMMON= $(INCLUDE) -DCHECKBLOCK=$(CHECKBLOCK) -DZERO_HALF=$(ZERO_HALF) -DZERO_FLOAT=$(ZERO_FLOAT)
LDFLAGS= $(LIBRARY) -lcudart  -lcurand -lstdc++

OBJ=HotspotExecute.o Log.o Parameters.o main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard *.h) Makefile

NVCCFLAGS+= 

all: obj  $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)  
	$(CXX) $(COMMON) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON)  $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(COMMON) $(ARCH)  $(NVCCFLAGS) -c $< -o $@

obj:
	mkdir -p obj


clean:
	rm -rf $(EXEC) $(OBJDIR)/*


SIMTIME=1000
#Cannot be bigger than 15 when BLOCK_SIZE is 32
#And 7 when BLOCK_SIZE is 16
PYRAMIDH=7


generate:
	./$(EXEC) -size=1024 -generate -verbose -sim_time=$(SIMTIME) \
		-input_temp=$(RADDIR)/data/hotspot/temp_1024 \
		-input_power=.$(RADDIR)/data/hotspot/power_1024 \
		-gold_temp=./gold_1024 -streams=10 \
		-redundancy=$(REDUNDANCY) -precision=$(PRECISION) \
		 -pyramid_height=$(PYRAMIDH)
		
test:
	./$(EXEC) -size=1024  -verbose -sim_time=$(SIMTIME) \
		-input_temp=$(RADDIR)/data/hotspot/temp_1024 \
		-input_power=$(RADDIR)/data/hotspot/power_1024 \
		-gold_temp=./gold_1024 -streams=10 -iterations=10 \
		-redundancy=$(REDUNDANCY) -precision=$(PRECISION) \
		 -pyramid_height=$(PYRAMIDH)
	
