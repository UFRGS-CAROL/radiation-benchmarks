LOGS=1
DEBUG?=0
PRECISION=double
REDUNDANCY=dmrmixed
CHECKBLOCK=0


CXX=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]	# Jetson Xavier

NVCCFLAGS= $(ARCH) -Xptxas -v -std=c++11
CXXFLAGS= -std=c++11 -fopenmp -O3
INCLUDE= -I/usr/local/cuda/include/ -I../../include/ -I../common/include/
LIBRARY= -L/usr/local/cuda/lib64/  -L../../include/ 

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

COMMON= $(INCLUDE) -DCHECKBLOCK=$(CHECKBLOCK)
LDFLAGS= $(LIBRARY) -lcudart -lcublas -lcurand -lstdc++

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
PYRAMIDH=10

generate:
	./$(EXEC) -size=1024 -generate -verbose -sim_time=$(SIMTIME) \
		-input_temp=../../../data/hotspot/temp_1024 \
		-input_power=../../../data/hotspot/power_1024 \
		-gold_temp=./gold_1024 -streams=10 \
		-redundancy=$(REDUNDANCY) -precision=$(PRECISION) \
		 -pyramid_height=$(PYRAMIDH)
		
test:
	./$(EXEC) -size=1024  -verbose -sim_time=$(SIMTIME) \
		-input_temp=../../../data/hotspot/temp_1024 \
		-input_power=../../../data/hotspot/power_1024 \
		-gold_temp=./gold_1024 -streams=10 -iterations=10 \
		-redundancy=$(REDUNDANCY) -precision=$(PRECISION) \
		 -pyramid_height=$(PYRAMIDH)
	
