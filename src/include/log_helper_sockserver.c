#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>

void *handle_client(void *sock) {
    char message[2000];
    char filename[500];
    char full_message[2500];
    int read;
    int socket = (int) sock;

    if(recv(socket , full_message , 2500 , 0) > 0){
	int i=0;
	while(full_message[i]!='|'&&i<500){
		filename[i]=full_message[i];
		i++;
	}
	if(i>499){
		close(socket);
		pthread_exit(NULL);
	}
	filename[i]='\0';
	i++;
	int j=0;
	while(full_message[i]!='\0'&&i<2500&&j<1999){
		message[j] = full_message[i];
		i++;
		j++;
	}
	message[j]='\0';
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
    close(socket);
    pthread_exit(NULL);
}

int main(int argc , char *argv[])
{
    int socket_desc , client_sock , c , read_size;
    struct sockaddr_in server , client;

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
