CUDAPATH=/usr/local/cuda
CC=gcc
CXX=g++
NVCC=$(CUDAPATH)/bin/nvcc
PRECISION=float
LOGS=1
USE_OMP=1

BUILDPROFILER=0
RADDIR=../../..

ITERATIONS=10
STREAMS=1
SIZE=1024
SIM_TIME=1000
DATA_DIR=$(RADDIR)/data/hotspot
DEF_INPUT_POWER=$(DATA_DIR)/power_$(SIZE)
DEF_INPUT_TEMP=$(DATA_DIR)/temp_$(SIZE)
DEF_GOLD=$(DATA_DIR)/gold_$(PRECISION)_$(SIZE)_$(SIM_TIME)

# Kepler, Volta
SMS= 35 53 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

LOGHELPER_INC=$(RADDIR)/libLogHelper/include/
LOGHELPER_LIB=$(RADDIR)/libLogHelper/

NVCCFLAGS= $(ARCH) -std=c++11
INCLUDE= -I$(CUDAPATH)/include -I$(RADDIR)/src/cuda/common/include  -I$(LOGHELPER_INC)
LIBRARY= -L$(CUDAPATH)/lib64/
CXXFLAGS= -I$(CUDAPATH)/include -L$(CUDAPATH)/lib64/ -lcudart  -std=c++11

TARGET=cuda_hotspot
RUN_SOURCES=./cuda_hotspot.cu


ifeq ($(LOGS), 1)
#INCLUDE+= -I$(LOGHELPER_INC)
NVCCFLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB) 
CXXFLAGS+= -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -DLOGS -lLogHelper 
endif

ifeq ($(USE_OMP), 1)
NVCCFLAGS+= -DUSE_OMP -Xcompiler " -fopenmp"
endif

ifeq ($(DEBUG), 1)
NVCCFLAGS+= -g -G
endif

ifeq ($(BUILDPROFILER), 1)
NVCCFLAGS+= -DBUILDPROFILER
LIBRARY+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) -ccbin $(CXX) $(INCLUDE) $(LIBRARY) $(NVCCFLAGS) $(RUN_SOURCES) -o $(TARGET)

clean:
	rm -rf $(TARGET)

generate: $(TARGET)
	./$(TARGET) -generate -size=$(SIZE) -input_power=$(DEF_INPUT_POWER) \
				-input_temp=$(DEF_INPUT_TEMP) -gold_temp=$(DEF_GOLD) \
				-sim_time=$(SIM_TIME) -verbose -iterations=1 \
				 -streams=$(STREAMS) -precision=$(PRECISION)

test: $(TARGET)
	./$(TARGET) -size=$(SIZE) -input_power=$(DEF_INPUT_POWER) \
				-input_temp=$(DEF_INPUT_TEMP) -gold_temp=$(DEF_GOLD) \
				-sim_time=$(SIM_TIME) -verbose -iterations=$(ITERATIONS) \
				 -streams=$(STREAMS) -precision=$(PRECISION)
				
