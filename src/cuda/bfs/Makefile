CXX=g++
EXEC=cudaBFS
LOGS=1
BUILDPROFILER=0
ITERATIONS=10

RADDIR=../../..
DATADIR=$(RADDIR)/data/bfs
GOLD=$(DATADIR)/gold.data
INPUT=$(DATADIR)/graph1MW_6.txt

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -Xptxas -v  -Wno-deprecated-gpu-targets
# NVCC optmizations flags
NVCCOPTFLAGS=
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 35 53 70 86
$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/libLogHelper/include -I$(RADDIR)/src/cuda/common/include

OBJDIR=./obj/
OBJ= bfs.o main.o Parameters.o kernel.o

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
LDFLAGS+= -L$(RADDIR)/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(EXEC)

$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE)

mkdir:
	mkdir -p $(OBJDIR)
	mkdir -p $(DATADIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)

generate:
	./$(EXEC) --input $(INPUT) --gold $(GOLD) --iterations 1 --verbose --generate
	
test:
	./$(EXEC) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose

