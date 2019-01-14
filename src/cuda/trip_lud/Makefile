CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
PRECISION?=float
DEF_SIZE?=1024
LOGS?=1
USE_OMP?=1
SAFE_MALLOC?=0

UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(UPPERCASED_PRECISION),HALF)
ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	# Tegra X1
ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif

NVCC_FLAGS= $(ARCH) -O3 -std=c++11
INCLUDE= -I/usr/local/cuda/include
LIBRARY= -L/usr/local/cuda/lib64/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11 -fopenmp

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

ifeq ($(LOGS), 1)
INCLUDE+= -I$(LOGHELPER_INC)
NVCC_FLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB) 
CXXFLAGS+= -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -DLOGS -lLogHelper 
endif

ifeq ($(USE_OMP), 1)
NVCC_FLAGS+= -DUSE_OMP -Xcompiler " -fopenmp -O3"
endif

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

ifeq ($(SAFE_MALLOC), 1)
NVCC_FLAGS+= -I../../include/safe_memory -DSAFE_MALLOC -lSafeMemory 
LIBRARY+= -L../../include/safe_memory
endif

ifeq ($(LOGS), 1)
INCLUDE+= -I$(LOGHELPER_INC)
NVCC_FLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB) 
CXXFLAGS+= -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -DLOGS -lLogHelper 
endif

OBJDIR=./obj/
OBJ=lud_kernel.o cuda_lud.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))

EXEC=$(PRECISION)_trip_lud

all: mkdir $(EXEC)

$(EXEC): $(OBJS) 
	$(CXX) $(COMMON)  $^ -o $@ $(LDFLAGS) $(CXXFLAGS)

$(OBJDIR)%.o: %.cpp
	$(CXX) $(COMMON)  $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu 
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) --compiler-options "$(CXXFLAGS)" -c $< -o $@
	
	
mkdir:
	mkdir -p ./obj

clean:
	rm -rf $(EXEC) ./obj/*

generate: $(GEN_EXEC)
	$(RUN_EXEC) -size=$(DEF_SIZE) -generate -input=$(DEF_INPUT) -gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	$(RUN_EXEC) -size=$(DEF_SIZE) -input=$(DEF_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose
