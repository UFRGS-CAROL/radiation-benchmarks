
#lib name, wraper file and output .so file
LIBRARY_NAME=log_helper
LIBRARY_NAME_WRAP=$(LIBRARY_NAME)_wrap

# log_helper include
INCLUDE=-I../

CXX = g++
CC = gcc
SWIG = swig

LUA_INCLUDE?=-I/usr/include/lua5.1
LUA_LIB?=-L/usr/lib/lua5.1

PYTHON_INCLUDE=/usr/include/python2.7/

CFLAGS=-fPIC

all:$(LIBRARY_NAME)_python $(LIBRARY_NAME)_lua


	

$(LIBRARY_NAME)_python: $(LIBRARY_NAME).c $(LIBRARY_NAME).i
	$(SWIG) -python $(LIBRARY_NAME).i
	$(CC)  $(CFLAGS) -c $(LIBRARY_NAME).c $(LIBRARY_NAME_WRAP).c -I$(PYTHON_INCLUDE) -I../
	$(CC) -shared $(LIBRARY_NAME).o $(LIBRARY_NAME_WRAP).o -o _$(LIBRARY_NAME).so


$(LIBRARY_NAME)_lua:
	$(SWIG) -lua $(LIBRARY_NAME).i
	$(CC) $(CFLAGS) $(LUA_INCLUDE) -c $(LIBRARY_NAME_WRAP).c -o $(LIBRARY_NAME_WRAP).o
	$(CC) $(CFLAGS) $(INCLUDE) -c $(LIBRARY_NAME).c -o  $(LIBRARY_NAME).o
	$(CC) -shared $(LUA_INCLUDE) $(LUA_LIB) $(LIBRARY_NAME_WRAP).o $(LIBRARY_NAME).o -o $(LIBRARY_NAME).so



clean:
	rm -rf *.o *py *so *pyc $(LIBRARY_NAME_WRAP).c
