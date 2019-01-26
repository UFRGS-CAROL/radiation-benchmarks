CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
CXXFLAGS= -std=c++11 -o3  -fPIC
EXEC=cudaCacheTest

VPATH=./src/
CUDAPATH=/usr/local/cuda

NVCCFLAGS= -std=c++11 -O3


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
INCLUDE= -I./src -I$(CUDAPATH)/include


OBJDIR=./obj/
OBJ= NVMLWrapper.o utils.o kernels.o main.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -G
endif

LDFLAGS= -L$(CUDAPATH)/lib64  -lcudart

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
	
test:
	./$(EXEC) /home/fernando/git_pesquisa/radiation-benchmarks/scripts/json_files/dgemm_check.json