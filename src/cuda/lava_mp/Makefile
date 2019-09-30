CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -pthread 
EXEC=cuda_lava

LOGS=0
BUILDPROFILER=0

VPATH=./src
#CUDAPATH=/home/carol/cuda/cuda-toolkit
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3 -Xptxas -v

PRECISION?=single
DATA_DIR=../../../data/lava
DEF_SIZE=20
STREAMS=10
CHARGES_INPUT=$(DATA_DIR)/lava_$(PRECISION)_charges_$(DEF_SIZE)
DISTANCES_INPUT=$(DATA_DIR)/lava_$(PRECISION)_distances_$(DEF_SIZE)
GOLD=$(DATA_DIR)/lava_$(PRECISION)_gold_$(DEF_SIZE)


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

OBJDIR=./obj/
OBJ= Parameters.o cuda_lava.o

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


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
generate: $(EXEC)
	./$(EXEC) -boxes=$(DEF_SIZE) -generate -streams=$(STREAMS) \
				-iterations=1 -verbose \
				-input_distances=$(DISTANCES_INPUT) \
				-input_charges=$(CHARGES_INPUT) 	\
				-output_gold=$(GOLD)

test: $(RUN_EXEC) generate
	./$(EXEC) -boxes=$(DEF_SIZE) -streams=$(STREAMS) \
				-iterations=10 -verbose \
				-input_distances=$(DISTANCES_INPUT) \
				-input_charges=$(CHARGES_INPUT) \
				-output_gold=$(GOLD)

	
	
