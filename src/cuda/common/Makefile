CXX=g++

MODE=cvr
CXXFLAGS= -shared
CUDAPATH=/usr/local/cuda
SRC=./src
LIB=./lib
CXXFLAGS+=-fPIC -Wall -Werror -Wextra -Wno-unknown-pragmas -std=c++11 -o3  -pthread


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
INCLUDE= -I$(CUDAPATH)/include -I./include

OBJ=JTX2Inst.o NVMLWrapper.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./include/*.h) Makefile

LIBS=JTX2Inst.so NVMLWrapper.so

all: clean $(OBJS) $(LIBS)

JTX2Inst.so: JTX2Inst.o 
	ar $(MODE) $(LIB)/JTX2Inst.so JTX2Inst.o 
	rm -f JTX2Inst.o

NVMLWrapper.so: NVMLWrapper.o
	ar $(MODE) $(LIB)/NVMLWrapper.so NVMLWrapper.o
	rm -f NVMLWrapper.o

%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS) $(INCLUDE)

clean:
	rm -rf $(LIBS) $(OBJ) lib/*.so
