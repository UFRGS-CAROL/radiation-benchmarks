#include<stdio.h>
#include<string.h>
#include<sys/socket.h>
#include<arpa/inet.h>
#include<unistd.h>
#include <pthread.h>

void *handle_client(void *sock) {
    char message[2000];
    char filename[500];
    int read;
    int socket = (int) sock;

    while( recv(socket , filename , 500 , 0) > 0 )
    {
        if ((read = recv(socket, message, 2000, 0)) > 1)
        {
            printf("filename: '%s'\n",filename);
            printf("message: '%s'\n",message);
	    FILE *fp = fopen(filename, "a+");
	    if(fp != NULL){
	    	fprintf(fp, "%s", message);
		fclose(fp);
	    }else{
	    	fprintf(stderr, "cannot open file %s\n",filename);
	    }
        }
        if (read == -1) {
            perror("Failed to receive message\n");
        }
        if (read == 0) {
            perror("Closing socket with unexpected way\n");
            break;
        }

    }
    pthread_exit(NULL);
}

int main(int argc , char *argv[])
{
    int socket_desc , client_sock , c , read_size;
    struct sockaddr_in server , client;
    char client_message[2000];

    //Create socket
    socket_desc = socket(AF_INET , SOCK_STREAM , 0);
    if (socket_desc == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");

    //Prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons( 8888 );

    //Bind
    if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
    {
        //print the error message
        perror("bind failed. Error");
        return 1;
    }
    puts("bind done");

    //Listen
    listen(socket_desc , 3);

    //Accept and incoming connection
    puts("Waiting for incoming connections...");
    c = sizeof(struct sockaddr_in);

    while (client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&c)) {

        if (client_sock < 0)
        {
            perror("accept failed");
            continue;
        }
        printf("incoming client\n");
        pthread_t thread;
        pthread_create(&thread,NULL, handle_client, (void*)client_sock);
    }
    return 0;
}
