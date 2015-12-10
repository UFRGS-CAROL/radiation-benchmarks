//#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <arpa/inet.h>

#define NUM_EXEC 2 //to reboot
#define NUM_THREADS 2
#define WAIT_SECONDS 15 //kill
#define NUM_EXEC_ACC 4 //to reboot
//#define ECC 10


//code to send signal to server that controls switch PING
#define PORT "3490" // the port client will be connecting to 
#define MAXDATASIZE 100 // max number of bytes we can get at once
#define SERVER_IP "192.168.1.5"
// get sockaddr, IPv4 or IPv6:

void *get_in_addr(struct sockaddr *sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}

	return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

void connect_server(){
	int sockfd, numbytes;  
	char buf[MAXDATASIZE];
	struct addrinfo hints, *servinfo, *p;
	int rv;
	char s[INET6_ADDRSTRLEN];


	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	if ((rv = getaddrinfo(SERVER_IP, PORT, &hints, &servinfo)) != 0) {
		fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
		return;
	}

	// loop through all the results and connect to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
				p->ai_protocol)) == -1) {
			perror("client: socket");
			continue;
		}

		if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
			close(sockfd);
			perror("client: connect");
			continue;
		}

		break;
	}

	if (p == NULL) {
		fprintf(stderr, "client: failed to connect\n");
		return;
	}

	inet_ntop(p->ai_family, get_in_addr((struct sockaddr *)p->ai_addr),
			s, sizeof s);
	printf("client: connecting to %s\n", s);

	freeaddrinfo(servinfo); // all done with this structure

	close(sockfd);
}



FILE* timefile;
FILE* logfile;
FILE* fileECC;

pthread_t thread[NUM_THREADS];

void get_time(){

	time_t timestamp = time(NULL);
	char time_s[50];
	sprintf(time_s, "%d", int(timestamp));

	char string[100] = "echo ";
	strcat(string, time_s);
	strcat(string, " > /home/carol/watchdog/timestamp.txt");
	system(string);

	printf("\n%s\n", string);
	//timefile = fopen("/home/carol/TestGPU/timestamp.txt", "w");
	//fprintf(timefile, "%d\n", (int)timestamp);
	//fclose(timefile);
}

void *GPUCall(void *id){

	//system("/home/carol/nupar-bench/CUDA/CCL/accl 8 4 /home/carol/nupar-bench/CUDA/CCL/Data/8Frames.pgm /home/carol/GOLD/accl_output_gold8_4 10000 &");
	
	//system("/home/carol/radiation-benchmarks/cuda/gemm/cudaGEMM 1024 /home/carol/GOLD/DGEMM_A_8192.matrix /home/carol/GOLD/DGEMM_B_8192.matrix /home/carol/GOLD/DGEMM_GOLD_1024.matrix 10000 &");
	
	        //system("/home/carol/radiation-benchmarks/cuda/gemm/cudaGEMM 2048 /home/carol/GOLD/DGEMM_A_8192.matrix /home/carol/GOLD/DGEMM_B_8192.matrix /home/carol/GOLD/DGEMM_GOLD_2048.matrix 10000 &");
	
//system("/home/carol/radiation-benchmarks/cuda/gemm/cudaGEMM 4096 /home/carol/GOLD/DGEMM_A_8192.matrix /home/carol/GOLD/DGEMM_B_8192.matrix /home/carol/GOLD/DGEMM_GOLD_4096.matrix 10000 &");

	//system("/home/carol/nupar-bench/CUDA/CCL/accl 8 4 /home/carol/nupar-bench/CUDA/CCL/Data/8Frames.pgm /home/carol/GOLD/accl_output_gold8_4 10000 &");
	

//	system("/home/carol/radiation-benchmarks/opencl/lavaMD/lavamd_check 19 4 /home/carol/radiation-benchmarks/opencl/lavaMD/kernel/kernel_gpu_opencl.cl 128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_distance_19_128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_charges_19_128 /home/carol/radiation-benchmarks/opencl/lavaMD/output_gold_19_128 1000000 &");
	//
	//system("/home/carol/radiation-benchmarks/opencl/lavaMD/lavamd_check 19 4 /home/carol/radiation-benchmarks/opencl/lavaMD/kernel/kernel_gpu_opencl.cl 128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_distance_19_128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_charges_19_128 /home/carol/radiation-benchmarks/opencl/lavaMD/output_gold_19_128 1000000 &");
	//
//	system("/home/carol/radiation-benchmarks/opencl/lavaMD/lavamd_check 23 4 /home/carol/radiation-benchmarks/opencl/lavaMD/kernel/kernel_gpu_opencl.cl 128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_distance_23_128 /home/carol/radiation-benchmarks/opencl/lavaMD/input_charges_23_128 /home/carol/radiation-benchmarks/opencl/lavaMD/output_gold_23_128 1000000 &");


	 system("/home/carol/nvidia2/run_jobs.sh");
//	system("/home/carol/run_clamr.sh");
	pthread_exit(NULL);
}

void *reboot(void* id){

	//connect
	connect_server();

	//get_time();

	int cont = 1;
	int cont_acc = 1;
	int contECC = 0;
	char lasttime1[20];
	int lasttime2;
	int i = 1;
	time_t timenow;
	time_t timeping;
	time_t timeECC;


	struct tm *ptm;
	char day[2], month[2], year[4], hour[2], second[2], minute[2];
		
	timenow = time(NULL);
	ptm = gmtime(&timenow);


	snprintf(day, sizeof(day + 1), "%d", ptm->tm_mday);
	snprintf(month, sizeof(month + 1), "%d", ptm->tm_mon+1);
	snprintf(year, sizeof(year + 1), "%d", ptm->tm_year+1900);
	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	
	
	while(1){
		//get_time();
		sleep(WAIT_SECONDS);

		//connect
		connect_server();

		timefile = fopen("/home/carol/watchdog/timestamp.txt", "r");	

		fgets (lasttime1, 20, timefile);
		fclose(timefile);

		lasttime2 = atoi(lasttime1);
		timenow = time(NULL);
		

		if (((int)timenow-lasttime2)>WAIT_SECONDS){

			logfile = fopen("/home/carol/kill_log.txt", "a");
			printf("Mais que %d segundos %d\n", WAIT_SECONDS, cont);

			//TIME OF KILL/////
			ptm = gmtime(&timenow);
			snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
			snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
			snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);	
			fprintf(logfile, "\n kill n. %d at %s/%s_%s:%s:%s", cont, day,month,hour,minute,second);
			

			fclose(logfile);

			if ((cont < NUM_EXEC)&&(cont_acc < NUM_EXEC_ACC)){

			//	system("sudo killall -9 lavamd_check");
			//	system("sudo killall -9 run_clamr.sh");
			//	system("sudo killall -9 clamr_gpuonly");
			                        system("sudo killall -9 run_jobs.sh");
					        system("sudo killall -9 sm_iadd");
					        system("sudo killall -9 sm_imad");
						system("sudo killall -9 sm_fadd");
						system("sudo killall -9 sm_ffma");
						system("sudo killall -9 sm_isetp");

						//system("sudo killall -9 sm_lrf");

				//sleep(2);
				//system("sudo nvidia-smi -i 0 -r &");

				sleep(2);				
				pthread_cancel(thread[0]);
				get_time();
				pthread_create(&thread[0],NULL, GPUCall, (void*)1);
				
			}
			else{
				
				logfile = fopen("/home/carol/kill_log.txt", "a");
			
				//TIME OF REBOOT/////
				timenow = time(NULL);
				ptm = gmtime(&timenow);
				snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
				snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);	
				
				fprintf(logfile, "\n reboot at %s/%s_%s:%s:%s", day,month,hour,minute,second);
	
				fclose(logfile);
				//CONNECT
				connect_server();
				system("sudo shutdown -r now");

				exit(0);
			}

			cont++;
			cont_acc++;
		}
		else
		{cont=0;}

		connect_server();
		
	
	}
	
	pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
	
	pthread_create(&thread[0],NULL, GPUCall, (void*)1);
	pthread_create(&thread[1],NULL, reboot, (void*)2);
	
	pthread_exit(NULL);
}





