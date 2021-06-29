DEBUG=0
#for radiation setup
LOGS=1
OMP=1
ALPHA=1.0
BETA=0.0
SIZE?=4096
ITERATIONS=1
PRECISION=float
CUBLAS=0
TENSOR_CORES=0
DMR=none
BUILDRELATIVEERROR=0
CHECKBLOCK=1

OUTPUTPREFIX=$(PRECISION)_$(ALPHA)_$(BETA)_$(SIZE)_cublas_$(CUBLAS)_tensor_$(TENSOR_CORES).matrix

# NVCC optmizations flags
NVCCOPTFLAGS=

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 35

# check for cuda 7.0 or lower
ifneq (, $(findstring cuda-7.0, $(CUDAPATH)))
else
# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]
# Tegra X2
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]
# Titan V | Tesla V100
SMS+= 53 70
# Xavier
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]
endif

$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))ARCH=-gencode arch=compute_35,code=[sm_35,compute_35]


VPATH=./src/
TARGET=gemm
OBJDIR=./obj/
CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc 
OPTS=-O3
LDFLAGS=  -L$(CUDAPATH)/lib64 -lcudart -lcurand -lcublas
COMMON= --std=c++11
CFLAGS= -Wall -Wfatal-errors

RAD_DIR=/home/carol/radiation-benchmarks

INCLUDE=-I$(CUDAPATH)/include -I../common  -I$(CUDAPATH)/include/

ifeq ($(CUBLAS), 1)
USECUBLAS= --use_cublas
endif

ifeq ($(TENSOR_CORES), 1)
USETENSORS= --tensor_cores
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
NVCCFLAGS+= -g -G -DDEBUG
endif

ifeq ($(BUILDRELATIVEERROR), 1)
NVCCFLAGS+= -DBUILDRELATIVEERROR=1
endif

ifeq ($(OMP), 1) 
CFLAGS+= -DOMP -fopenmp
NVCCFLAGS+= -Xcompiler "-DOMP -fopenmp"
endif

ifeq ($(LOGS), 1)
INCLUDE+=-I../../include/
NVCCFLAGS+= -DLOGS
CFLAGS+= -DLOGS
LDFLAGS+= -L../../include/ -lLogHelper -DLOGS=1
endif


OBJ= main.o setup_gemm.o Parameters.o common.o setup_tensor_cores.o
CFLAGS+=$(OPTS)

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: obj $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(INCLUDE) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(COMMON) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@ 


obj:
	mkdir -p obj


.PHONY: clean
clean:
	rm -rf $(OBJS) $(TARGET)

# GEMM options
generate:
	./$(TARGET) --size $(SIZE) --generate --verbose \
			--precision $(PRECISION) --dmr none \
			--alpha $(ALPHA) --beta $(BETA) \
			--input_a a_$(OUTPUTPREFIX) \
			--input_b b_$(OUTPUTPREFIX) \
			--input_c c_$(OUTPUTPREFIX) \
			--gold  g_$(OUTPUTPREFIX) $(USECUBLAS) $(USETENSORS)
	
test:
	./$(TARGET) --size $(SIZE) --verbose \
			--precision $(PRECISION) --dmr $(DMR) \
			--iterations $(ITERATIONS)  \
			--alpha $(ALPHA) --beta $(BETA) \
			--input_a a_$(OUTPUTPREFIX) \
			--input_b b_$(OUTPUTPREFIX) \
			--input_c c_$(OUTPUTPREFIX) \
			--gold  g_$(OUTPUTPREFIX)	\
			--opnum $(CHECKBLOCK) $(USECUBLAS) $(USETENSORS)
			
