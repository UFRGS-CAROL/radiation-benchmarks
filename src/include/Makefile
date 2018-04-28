# make static lib

CC=gcc

CCFLAGS=-fPIC -Wall -Werror -Wextra -shared -Wno-unknown-pragmas

OBJ=log_helper.o
LIBOUT=libLogHelper.so

all: clean log_helper.c
	$(CC) -c $(CCFLAGS) log_helper.c -o $(OBJ)
	ar cvr $(LIBOUT) $(OBJ)


client: clean log_helper_sockclient.c
	$(CC) -c  $(CCFLAGS) log_helper_sockclient.c -o $(OBJ)
	ar cvr $(LIBOUT) $(OBJ)

server: log_helper_sockserver.c
	$(CC) log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf $(LIBOUT) *.o *.a server
