CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 
EXEC=cudaMicro
LOGS=1
BUILDPROFILER=0
MICRO=pythagorean
ITERATIONS=10
OPNUM=100000
PRECISION=float
FASTMATH=0

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -O3 -Xptxas -v

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

OBJDIR=./obj/
OBJ=  micro_real_kernels.o micro_ldst_branch_kernels.o 
OBJ+= micro_int_kernels.o main.o Parameters.o utils.o 

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

ifeq ($(FASTMATH), 1)
USEFASTMATH= --fast_math
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
	
	
generate:
	./$(EXEC) --iterations $(ITERATIONS) --inst $(MICRO) --verbose  \
			  --opnum $(OPNUM) --precision $(PRECISION) $(USEFASTMATH) \
			  --generate --gold ./gold_$(MICRO).data
			  
test:
	./$(EXEC) --iterations $(ITERATIONS) --inst $(MICRO) --verbose  \
			  --opnum $(OPNUM) --precision $(PRECISION) $(USEFASTMATH) \
			  --gold ./gold_$(MICRO).data
	
