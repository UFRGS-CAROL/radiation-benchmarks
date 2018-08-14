CC=gcc
CXX=g++

CCFLAGS=-fPIC -Wall -Werror -Wextra -shared -Wno-unknown-pragmas
CXXFLAGS=-std=c++11 -fPIC -Wall -Werror -Wextra -shared -Wno-unknown-pragmas -lc

LOGHELPER=log_helper.o
LOGHELPERWRAPPER=log_helper_wrapper.o

OBJ= $(LOGHELPER) $(LOGHELPERWRAPPER)

LIBOUT=libLogHelper.so
LIBOUTA=libLogHelper.a

LIB=lib

SRCCXX=lib/log_helper.cpp
SRCCC=lib/log_helper.c
SRCWRAPPER=lib/log_helper_wrapper.cpp

INCLUDE=-I./

all: clean cxx cc $(SRCCXX) $(SRCWRAPPER)


cc: 
	$(CXX) -c $(CXXFLAGS) $(SRCCXX) $(INCLUDE)
	$(CXX) -c $(CXXFLAGS) $(SRCWRAPPER) $(INCLUDE)
	ar cvr $(LIB)/$(LIBOUT) $(OBJ)
	rm $(OBJ)

cxx: 
	$(CXX) -c $(CXXFLAGS) $(SRCCXX) $(INCLUDE)
	ar cvr $(LIB)/$(LIBOUT) $(LOGHELPER)
	rm $(LOGHELPER)


client: clean log_helper_sockclient.c
	$(CC) -c  $(CCFLAGS) log_helper_sockclient.c -o $(OBJ)
	ar cvr $(LIBOUT) $(OBJ)

server: log_helper_sockserver.c
	$(CC) log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf $(LIB)/$(LIBOUT) $(LIB)/*.o $(LIB)/*.a server