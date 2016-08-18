#MPICH_DIR=/usr/include/mpich
#MPI_FLAGS = -DUSE_MPI -I$(MPICH_DIR)/include

NVCC		= nvcc -ccbin /usr/bin/gcc
FLAGS		= -arch=sm_35
DFLAGS	= $(MPI_FLAGS) -lineinfo
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG 

TARGET=lulesh

ifeq ($(dbg), 1)
	DEBUG_FLAG= -g -G
endif

LOG_FLAGS=
ifeq ($(logs), 1)
	LOGHELPER_INC=../../include/
	LOGHELPER_LIB=../../include/
	LOG_FLAGS += -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS -std=c++11 
endif 


#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

#LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib 
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

#all: release 
all: $(TARGET) #generate


debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

$(TARGET): allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(NVCC) $(LOG_FLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o $(TARGET)

allocator.o: allocator.cu vector.h
	$(NVCC) $(FLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(NVCC) $(LINKFLAGS) $(LOG_FLAGS) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

lulesh-comms.o: lulesh-comms.cu
	$(NVCC) $(FLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(NVCC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o


generate:
	 ./lulesh -s 50 -i 1 -g 1 gold_50

test:
	 ./lulesh -s 50 -i 1 -g 0 gold_50
	
clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt
