all: server client


server: server.c
	gcc server.c -o server -lpthread -w

client: client.c
	gcc client.c -o client

clean:
	rm server client
