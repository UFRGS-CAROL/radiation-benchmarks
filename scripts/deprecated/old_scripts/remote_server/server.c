/*
** To choose what devices to test, see function choose_tests() line 65
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>


#define PORT "3490"  // the port users will be connecting to
#define BACKLOG 20	 // how many pending connections queue will hold

#define MAX_TIME 95  // time to reboot (seconds)
#define MAX_TIME_LAST_CONN 180// time to reboot in case the boot fails (seconds)

#define CAROLXEON1 "192.168.1.6" //ID = 0
#define CAROLXEON2 "192.168.1.7" //ID = 1
#define CAROLK20 "192.168.1.8" //ID = 2
#define CAROLK40 "192.168.1.9" //ID = 3
#define CAROLAPU1 "192.168.1.10" //ID = 4
#define CAROLAPU2 "192.168.1.11" //ID = 5
#define CAROLK1A "192.168.1.12" //ID = 6
#define CAROLK1B "192.168.1.13" //ID = 7
#define CAROLK1C "192.168.1.14" //ID = 8

//TODO: define switch ip and port of each device

//OK
#define CAROLXEON1_SW_ON  "./switch9258.py 4 On  192.168.1.100"
#define CAROLXEON1_SW_OFF "./switch9258.py 4 Off 192.168.1.100"
//OK
#define CAROLXEON2_SW_ON  "./switch9258.py 4 On  192.168.1.102"
#define CAROLXEON2_SW_OFF "./switch9258.py 4 Off 192.168.1.102"
//TODO
#define CAROLK20_SW_ON    "./switch9258.py 1 On  192.168.1.105"
#define CAROLK20_SW_OFF   "./switch9258.py 1 Off 192.168.1.105"
//OK
#define CAROLK40_SW_ON    "./switch9258.py 3 On  192.168.1.101"
#define CAROLK40_SW_OFF   "./switch9258.py 3 Off 192.168.1.101"
//OK
#define CAROLAPU1_SW_ON   "./switch9258.py 1 On  192.168.1.100"
#define CAROLAPU1_SW_OFF  "./switch9258.py 1 Off 192.168.1.100"
//OK
#define CAROLAPU2_SW_ON   "./switch9258.py 1 On  192.168.1.101"
#define CAROLAPU2_SW_OFF  "./switch9258.py 1 Off 192.168.1.101"
//OK
#define CAROLK1A_SW_ON    "./switch9258.py 3 On  192.168.1.100"
#define CAROLK1A_SW_OFF   "./switch9258.py 3 Off 192.168.1.100"
//OK
#define CAROLK1B_SW_ON    "./switch9258.py 2 On  192.168.1.102"
#define CAROLK1B_SW_OFF   "./switch9258.py 2 Off 192.168.1.102"
//OK
#define CAROLK1C_SW_ON    "./switch9258.py 3 On  192.168.1.102"
#define CAROLK1C_SW_OFF   "./switch9258.py 3 Off 192.168.1.102"
//OK
#define FPGA_SW_ON        "./switch9258.py 2 On  192.168.1.100"
#define FPGA_SW_OFF       "./switch9258.py 2 Off 192.168.1.100"


time_t carol[9];
time_t last_conn[9]; //time from the last connection
int test_carol[9];
int carol_reboot_count[9]={0,0,0,0,0,0,0,0,0};

void choose_tests(){
	test_carol[0] = 0; // testar carolxeon1? 1=yes 0=no
	test_carol[1] = 1; // testar carolxeon2? 1=yes 0=no
	test_carol[2] = 0; // testar carolk20? 1=yes 0=no
	test_carol[3] = 1; // testar carolk40? 1=yes 0=no
	test_carol[4] = 0; // testar carolapu1? 1=yes 0=no
	test_carol[5] = 1; // testar carolapu2? 1=yes 0=no
	test_carol[6] = 1; // testar carolk1a? 1=yes 0=no
	test_carol[7] = 0; // testar carolk1b? 1=yes 0=no
	test_carol[8] = 0; // testar carolk1c? 1=yes 0=no
}

void write_file(char *msg, int carolID){

	FILE * fp;
	system("date >> server_log.txt");

	char msg_final[150];
	//sprintf(msg_final, "%s%d\n", msg, carolID);
	if(carolID==0)
		sprintf(msg_final,"%s,carolxeon1\n",msg);
	else if(carolID==1)
		sprintf(msg_final,"%s,carolxeon2\n",msg);
	else if(carolID==2)
		sprintf(msg_final,"%s,carolk20\n",msg);
	else if(carolID==3)
		sprintf(msg_final,"%s,carolk40\n",msg);
	else if(carolID==4)
		sprintf(msg_final,"%s,carolapu1\n",msg);
	else if(carolID==5)
		sprintf(msg_final,"%s,carolapu2\n",msg);
	else if(carolID==6)
		sprintf(msg_final,"%s,carolk1a\n",msg);
	else if(carolID==7)
		sprintf(msg_final,"%s,carolk1b\n",msg);
	else if(carolID==8)
		sprintf(msg_final,"%s,carolk1c\n",msg);

	fp = fopen("server_log.txt", "a");
	fprintf(fp, "%s\n", msg_final);
	fclose(fp);
}

void *thread_carol(void *threadid)
{
   int i;
   time_t now;

   printf("Thread Handling Carol Machines Created\n");
   while(1){
	sleep(20);
	//printf("thread sleep\n");

for(i = 0; i < 9; i++){
		
		// if we dont want to test this device we jump to the next one
		if(test_carol[i] != 1){
			continue;
		}
		if(carol_reboot_count[i]>10){
			test_carol[i]=0;
			write_file("Tired of rebooting carol ",i);
		}

		time(&now);
		if(carol[i] != NULL){
			//printf("carol[%d] != NULL\n", i);
			if(difftime(now, carol[i]) > MAX_TIME){
				carol_reboot_count[i]++;
				printf("\n\n\e[31mreboot carol%d\n",i);
				printf("\e[0m");
				write_file("Reboot carol", i);
				if(i == 0){ //carolxeon1
					system(CAROLXEON1_SW_OFF);
					sleep(20);
					system(CAROLXEON1_SW_ON);
				}
				if(i == 1){ //carolxeon2
					system(CAROLXEON2_SW_OFF);
					sleep(20);
					system(CAROLXEON2_SW_ON);
				}
				if(i == 2){ //carolk20
					system(CAROLK20_SW_OFF);
					sleep(20);
					system(CAROLK20_SW_ON);
				}
				if(i == 3){ //carolk40
					system(CAROLK40_SW_OFF);
					sleep(20);
					system(CAROLK40_SW_ON);
				}
				if(i == 4){ //carolapu1
					system(CAROLAPU1_SW_OFF);
					sleep(20);
					system(CAROLAPU1_SW_ON);
				}
				if(i == 5){ //carolapu2
					system(CAROLAPU2_SW_OFF);
					sleep(20);
					system(CAROLAPU2_SW_ON);
				}
				if(i == 6){ //carolk1a
					system(CAROLK1A_SW_OFF);
					system(FPGA_SW_OFF);
					sleep(20);
					system(CAROLK1A_SW_ON);
					system(FPGA_SW_ON);
				}
				if(i == 7){ //carolk1b
					system(CAROLK1B_SW_OFF);
					system(FPGA_SW_OFF);
					sleep(20);
					system(CAROLK1B_SW_ON);
					system(FPGA_SW_ON);
				}
				if(i == 8){ //carolk1c
					system(CAROLK1C_SW_OFF);
					system(FPGA_SW_OFF);
					sleep(20);
					system(CAROLK1C_SW_ON);
					system(FPGA_SW_ON);
				}
/*
				if(i == 0){//carol1
					system(CAROL1_SW_OFF);
					sleep(20);
					system(CAROL1_SW_ON);
				}
				if(i == 1){//carol3
					system(CAROL3_SW_OFF);
					sleep(20);
					system(CAROL3_SW_ON);
				}
				if(i == 2){//carolAPU1
					system(CAROLAPU1_SW_OFF);
					sleep(20);
					system(CAROLAPU1_SW_ON);
				}
*/

				carol[i] = NULL;
			}
		}
		if( difftime(now, last_conn[i]) > MAX_TIME_LAST_CONN ){
			carol_reboot_count[i]++;
			printf("\n\n\e[31mboot problem for carolID=%d\n",i);
			printf("\e[0m");
			write_file("Boot problem for carol", i);
			if(i == 0){ //carolxeon1
				system(CAROLXEON1_SW_OFF);
				sleep(20);
				system(CAROLXEON1_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 1){ //carolxeon2
				system(CAROLXEON2_SW_OFF);
				sleep(20);
				system(CAROLXEON2_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 2){ //carolk20
				system(CAROLK20_SW_OFF);
				sleep(20);
				system(CAROLK20_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 3){ //carolk40
				system(CAROLK40_SW_OFF);
				sleep(20);
				system(CAROLK40_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 4){ //carolapu1
				system(CAROLAPU1_SW_OFF);
				sleep(20);
				system(CAROLAPU1_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 5){ //carolapu2
				system(CAROLAPU2_SW_OFF);
				sleep(20);
				system(CAROLAPU2_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 6){ //carolk1a
				system(CAROLK1A_SW_OFF);
				system(FPGA_SW_OFF);
				sleep(20);
				system(CAROLK1A_SW_ON);
				system(FPGA_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 7){ //carolk1b
				system(CAROLK1B_SW_OFF);
				system(FPGA_SW_OFF);
				sleep(20);
				system(CAROLK1B_SW_ON);
				system(FPGA_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 8){ //carolk1c
				system(CAROLK1C_SW_OFF);
				system(FPGA_SW_OFF);
				sleep(20);
				system(CAROLK1C_SW_ON);
				system(FPGA_SW_ON);
				time(&last_conn[i]);
			}
/*
			if(i == 0){//carol1
				system(CAROL1_SW_OFF);
				sleep(20);
				system(CAROL1_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 1){//carol3
				system(CAROL3_SW_OFF);
				sleep(20);
				system(CAROL3_SW_ON);
				time(&last_conn[i]);
			}
			if(i == 2){//carol3
				system(CAROLAPU1_SW_OFF);
				sleep(20);
				system(CAROLAPU1_SW_ON);
				time(&last_conn[i]);
			}
*/

			carol[i] = NULL;
		}
	}
   }
   pthread_exit(NULL);
}


void sigchld_handler(int s)
{
	while(waitpid(-1, NULL, WNOHANG) > 0);
}

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}

	return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int main(void)
{
	int sockfd, new_fd;  // listen on sock_fd, new connection on new_fd
	struct addrinfo hints, *servinfo, *p;
	struct sockaddr_storage their_addr; // connector's address information
	socklen_t sin_size;
	struct sigaction sa;
	int yes=1;
	char s[INET6_ADDRSTRLEN];
	int rv;

	choose_tests();
	int i;
	for(i=0;i<9;i++){
		printf("test_carol[%d]=%d\n",i,test_carol[i]);
	}

	int rc;
	pthread_t thread;
	printf("creating thread that handles reboot\n");
	rc = pthread_create(&thread, NULL, thread_carol, NULL);
	if(rc){
		printf("ERROR; return code from pthread_create() is %d\n", rc);
	        exit(-1);
	}


	if(test_carol[0]) system(CAROLXEON1_SW_ON);
	if(test_carol[1]) system(CAROLXEON2_SW_ON);
	if(test_carol[2]) system(CAROLK20_SW_ON);
	if(test_carol[3]) system(CAROLK40_SW_ON);
	if(test_carol[4]) system(CAROLAPU1_SW_ON);
	if(test_carol[5]) system(CAROLAPU2_SW_ON);
	if(test_carol[6]) system(CAROLK1A_SW_ON); system(FPGA_SW_ON);
	if(test_carol[7]) system(CAROLK1B_SW_ON); system(FPGA_SW_ON);
	if(test_carol[8]) system(CAROLK1C_SW_ON); system(FPGA_SW_ON);


	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE; // use my IP

	if ((rv = getaddrinfo(NULL, PORT, &hints, &servinfo)) != 0) {
		fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
		return 1;
	}

	// loop through all the results and bind to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
				p->ai_protocol)) == -1) {
			perror("server: socket");
			continue;
		}

		if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
				sizeof(int)) == -1) {
			perror("setsockopt");
			exit(1);
		}

		if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
			close(sockfd);
			perror("server: bind");
			continue;
		}

		break;
	}

	if (p == NULL)  {
		fprintf(stderr, "server: failed to bind\n");
		return 2;
	}

	freeaddrinfo(servinfo); // all done with this structure

	if (listen(sockfd, BACKLOG) == -1) {
		perror("listen");
		exit(1);
	}

	sa.sa_handler = sigchld_handler; // reap all dead processes
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART;
	if (sigaction(SIGCHLD, &sa, NULL) == -1) {
		perror("sigaction");
		exit(1);
	}

	printf("server: waiting for connections...\n");
	// set the initial time for last connections array
	time(&last_conn[0]);
	time(&last_conn[1]);
	time(&last_conn[2]);
	time(&last_conn[3]);
	time(&last_conn[4]);
	time(&last_conn[5]);
	time(&last_conn[6]);
	time(&last_conn[7]);
	time(&last_conn[8]);

	int count_conn = 0;
	while(1) {  // main accept() loop
		sin_size = sizeof their_addr;
		new_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
		if (new_fd == -1) {
			perror("accept");
			continue;
		}

		inet_ntop(their_addr.ss_family,
			get_in_addr((struct sockaddr *)&their_addr),
			s, sizeof s);
		//printf("server: got connection from %s\n", s);

		if(strcmp(s, CAROLXEON1) == 0){
			printf("\e[32m%d-Connection from CAROLXEON1\n",count_conn);
			time(&carol[0]);
			time(&last_conn[0]);
		} else if(strcmp(s, CAROLXEON2) == 0){
			printf("\e[32m%d-Connection from CAROLXEON2\n",count_conn);
			time(&carol[1]);
			time(&last_conn[1]);		
		} else if(strcmp(s, CAROLK20) == 0){
			printf("\e[34m%d-Connection from CAROLK20\n",count_conn);
			time(&carol[2]);
			time(&last_conn[2]);		
		} else if(strcmp(s, CAROLK40) == 0){
			printf("\e[34m%d-Connection from CAROLK40\n",count_conn);
			time(&carol[3]);
			time(&last_conn[3]);		
		} else if(strcmp(s, CAROLAPU1) == 0){
			printf("\e[37m%d-Connection from CAROLAPU1\n",count_conn);
			time(&carol[4]);
			time(&last_conn[4]);		
		} else if(strcmp(s, CAROLAPU2) == 0){
			printf("\e[37m%d-Connection from CAROLAPU2\n",count_conn);
			time(&carol[5]);
			time(&last_conn[5]);		
		} else if(strcmp(s, CAROLK1A) == 0){
			printf("\e[39m%d-Connection from CAROLK1A\n",count_conn);
			time(&carol[6]);
			time(&last_conn[6]);		
		} else if(strcmp(s, CAROLK1B) == 0){
			printf("\e[39m%d-Connection from CAROLK1B\n",count_conn);
			time(&carol[7]);
			time(&last_conn[7]);		
		} else if(strcmp(s, CAROLK1C) == 0){
			printf("\e[39m%d-Connection from CAROLK1C\n",count_conn);
			time(&carol[8]);
			time(&last_conn[8]);		
		}
		printf("\e[0m");
		count_conn++;
		close(new_fd);  // parent doesn't need this
	}

	return 0;
}

