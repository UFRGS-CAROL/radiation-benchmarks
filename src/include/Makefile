# make static lib

CC=gcc

OBJ=log_helper.o

all: clean log_helper.c
	$(CC) -c log_helper.c -o $(OBJ)
	ar cvr libLogHelper.a $(OBJ)


client: clean log_helper_sockclient.c
	$(CC) -c log_helper_sockclient.c -o $(OBJ)
	ar cvr libLogHelper.a $(OBJ)

server: log_helper_sockserver.c
	$(CC) log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf libLogHelper.a *.o *.a server
