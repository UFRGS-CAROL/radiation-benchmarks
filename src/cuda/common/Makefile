CXX=g++

MODE=cvr
CXXFLAGS= -shared
CUDAPATH=/usr/local/cuda
SRC=./src
JTX2SRC=$(SRC)/jtx2/src
LIB=./lib
CXXFLAGS+=-fPIC -Wall -Wextra -Wno-unknown-pragmas -std=c++11 -o3  -pthread -li2c


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
INCLUDE= -I$(CUDAPATH)/include -I./include

JTX2OBJ= jtx1pow.o jtx1rate.o jtx1temp.o
OBJ= $(JTX2OBJ) JTX2Inst.o NVMLWrapper.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./include/*.h ./src/include/*.h) Makefile

LIBS=JTX2Inst.so NVMLWrapper.so

all: clean $(OBJS) $(LIBS)

JTX2Inst.so: JTX2Inst.o 
	ar $(MODE) $(LIB)/JTX2Inst.so JTX2Inst.o $(JTX2OBJ)
	rm -f JTX2Inst.o

NVMLWrapper.so: NVMLWrapper.o
	ar $(MODE) $(LIB)/NVMLWrapper.so NVMLWrapper.o
	rm -f NVMLWrapper.o

%.o:  $(JTX2SRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS) $(INCLUDE)

%.o: $(SRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS) $(INCLUDE)

clean:
	rm -rf $(LIBS) $(OBJ) lib/*.so
