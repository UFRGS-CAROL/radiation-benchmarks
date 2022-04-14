CXX=g++
EXEC=cudaCFD
LOGS=1
BUILDPROFILER=0
DEBUG=0
ITERATIONS=10
STREAMS=1

VPATH=./src
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
RADDIR=../../..
DATADIR=$(RADDIR)/data/cfd
INPUT_BASE=missile.domn.0.2M
INPUT=$(DATADIR)/$(INPUT_BASE)
GOLD=$(DATADIR)/cfd_gold_$(INPUT_BASE)

NVCCFLAGS= -std=c++11 -Xptxas -v -Xcompiler -fopenmp  -Xcompiler -fPIC
CXXFLAGS= -std=c++11 -O3 -fPIC -pthread -fopenmp

INCLUDE= -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/libLogHelper/include
INCLUDE+= -I$(RADDIR)/src/cuda/common/include
# Kepler, Volta
SMS= 35 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/libLogHelper  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
NVCCFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJDIR=./obj/
OBJ= euler3d.o main.o Parameters.o

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

clean:
	rm -f $(OBJDIR)* $(EXEC)

generate:
	./$(EXEC) --streams $(STREAMS) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose --generate

test:
	./$(EXEC) --streams $(STREAMS) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose
	
