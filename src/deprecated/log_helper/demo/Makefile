CXX=g++
CC=gcc


LIB=-L../lib -lLogHelper
INC=-I../include

CXXFLAGS=$(LIB) $(INC)
CCFLAGS=$(LIB) $(INC)

all: cpp cc 


cpp:
	$(CXX) example_log.cpp $(CXXFLAGS) -o example_log_cpp

cc:
	$(CC) example_log.c $(CCFLAGS) -o example_log_c
	
clean:
	rm -rf example_log_cpp example_log_c