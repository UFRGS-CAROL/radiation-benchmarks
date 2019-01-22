STATIC?=0
CC=gcc

LIBOUT=libLogHelper.so
MODE=cvr
CCFLAGS= -shared


# make static lib
ifeq ($(STATIC), 1)
LIBOUT=libLogHelper.a
CCFLAGS=
MODE=rcs
endif


CCFLAGS+=-fPIC -Wall -Werror -Wextra -Wno-unknown-pragmas


OBJ=log_helper.o

all: clean log_helper.c
	$(CC) -c $(CCFLAGS) log_helper.c -o $(OBJ)
	ar $(MODE) $(LIBOUT) $(OBJ)


client: clean log_helper_sockclient.c
	$(CC) -c  $(CCFLAGS) log_helper_sockclient.c -o $(OBJ)
	ar cvr $(LIBOUT) $(OBJ)

server: log_helper_sockserver.c
	$(CC) log_helper_sockserver.c -o server -lpthread

clean:
	rm -rf $(LIBOUT) *.o *.a server
