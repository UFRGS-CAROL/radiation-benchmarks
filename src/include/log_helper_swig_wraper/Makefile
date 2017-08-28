
#lib name, wraper file and output .so file
LIBRARY_NAME=log_helper
LIBRARY_NAME_WRAP=$(LIBRARY_NAME)_wrap
LIBRARY_NAME_SO=_$(LIBRARY_NAME)


CXX = g++
CC = gcc
SWIG = swig

PYTHON_INCLUDE=/usr/include/python2.7/


all:$(LIBRARY_NAME)_python $(LIBRARY_NAME)_lua


	

$(LIBRARY_NAME)_python: $(LIBRARY_NAME).c $(LIBRARY_NAME).i
	$(SWIG) -python $(LIBRARY_NAME).i
	$(CC)  -fPIC -c $(LIBRARY_NAME).c $(LIBRARY_NAME_WRAP).c -I$(PYTHON_INCLUDE) -I../
	$(CC) -shared $(LIBRARY_NAME).o $(LIBRARY_NAME_WRAP).o -o LIBRARY_NAME_SO.so


$(LIBRARY_NAME)_lua:
	echo "not yet"



clean:
	rm -rf *.o *py *so *pyc
