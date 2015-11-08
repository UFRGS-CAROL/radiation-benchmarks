#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

//=======================local log helper
// Location of timetamp file for software watchdog
//char timestamp_watchdog_local[200] = "/home/carol/watchdog/timestamp.txt";
char *timestamp_watchdog_local;
char timestamp_file_local[] = "timestamp.txt";
char vardir_key_local[]="vardir";
// Max errors that can be found for a single iteration
// If more than max errors is found, exit the program
unsigned long int max_errors_per_iter_local = 500;
char config_file_local[]="/etc/radiation-benchmarks.conf";
char *absolute_path;
char logdir_key[]="logdir";
int log_error_detail_count = 0;
// Used to print the log only for some iterations, equal 1 means print every iteration
int iter_interval_print_local = 1;
char log_file_name_local[200] = "";
char full_log_file_name_local[300] = "";

// Saves the last amount of error found for a specific iteration
unsigned long int last_iter_errors_local = 0;
// Saves the last iteration index that had an error
unsigned long int last_iter_with_errors_local = 0;

unsigned long int kernels_total_errors_local = 0;
unsigned long int iteration_number_local = 0;
double kernel_time_local_acc_local = 0;
double kernel_time_local = 0;
long long it_time_start_local;
// ~ ===========================================================================
inline long long get_time() {
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000) + tv.tv_usec;
};

// ~ ===========================================================================
unsigned long int set_max_errors_iter_local(unsigned long int max_errors){
     max_errors_per_iter_local = max_errors;

     return max_errors_per_iter_local;
};

// ~ ===========================================================================
// Set the interval the program must print log details, default is 1 (each iteration)
int set_iter_interval_print_local(int interval){
    if(interval < 1) {
        iter_interval_print_local = 1;
    }
    else {
        iter_interval_print_local = interval;
    }

    return iter_interval_print_local;
};

// ~ ===========================================================================
// Update with current timestamp the file where the software watchdog watchs
void update_timestamp_local() {
    time_t timestamp = time(NULL);
    char time_s[50];
    char string[100] = "echo ";

    sprintf(time_s, "%d", (int) timestamp);

    strcat(string, time_s);
    strcat(string, " > ");
    strcat(string, timestamp_watchdog_local);
    int res = system(string);
};

// ~ ===========================================================================
// Read config file to get the value of a 'key = value' pair
char * getValueConfig_local(char * key){
	FILE * fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char value[200];
	int i,j;
	int key_not_match;
	
	fp = fopen(config_file_local, "r");
	if (fp == NULL)
		return NULL;
	
	while ((read = getline(&line, &len, fp)) != -1) {
		// ignore comments and sections in config file
		if(line[0] == '#' || line[0] == '[')
			continue;

		// remove white spaces
		for(i=0;line[i]==' '; i++);
		// check if key of this line is the key we are looking for
		j=0;
		key_not_match=0;
		for(; line[i]!= ' ' && line[i] != '=' && key[j]!= '\0'; i++){
			if(key[j]!=line[i]){
				key_not_match=1;
				break;
			}
			j++;
		}
		// Key not matched
		if(key_not_match)
			continue;
		// key of line is a substring of the key we are looking for
		if(key[j]!='\0')
			continue;
		// key matched but is a substring of current key
		if(line[i] !=' ' && line[i]!='=')
			continue;
		// ignore spaces and '=' to go the the frist character of value
		for(;line[i]==' '||line[i] == '='; i++);
		j=0;
		// copy value to buffer until end of line or '#' is found
		for(;line[i]!='\0'&&line[i]!='#'&&line[i]!='\n';i++){
			value[j]=line[i];
			j++;
		}
		value[j]='\0';
		char *v = (char *)malloc(sizeof(char)*strlen(value)+2);
		strcpy(v, value);
		fclose(fp);
		if (line)
			free(line);
		return v;
	}
	
	fclose(fp);
	if (line)
		free(line);
	return NULL;
};

// ~ ===========================================================================
// Return the name of the log file generated
char * get_log_file_name_local_local(){
    return full_log_file_name_local;
};

// ~ ===========================================================================
// Generate the log file name, log info from user about the test to be executed and reset log variables
int start_log_file_local(char *benchmark_name, char *test_info){

    char *var_dir=getValueConfig_local(vardir_key_local);
    if(!var_dir){
        fprintf(stderr, "[ERROR] Could not read var dir in config file '%s'\n",config_file_local);
	return 1;//exit(1);
    }
    timestamp_watchdog_local = (char *)malloc(sizeof(char)* (strlen(var_dir)+strlen(timestamp_file_local)+4) );
    strcpy(timestamp_watchdog_local, var_dir);
    if(strlen(timestamp_watchdog_local) > 0 && timestamp_watchdog_local[strlen(timestamp_watchdog_local)-1] != '/' )
        strcat(timestamp_watchdog_local, "/");
    strcat(timestamp_watchdog_local, timestamp_file_local);
    
    update_timestamp_local();

    time_t file_time;
    struct tm *ptm;
    char day[10], month[10], year[15], hour[10], second[10], minute[10];
    char log_file_name_local[180] = "";

    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day,       sizeof(day),    "%02d", ptm->tm_mday);
    snprintf(month,     sizeof(month),  "%02d", ptm->tm_mon+1);
    snprintf(year,      sizeof(year),   "%04d", ptm->tm_year+1900);
    snprintf(hour,      sizeof(hour),   "%02d", ptm->tm_hour);
    snprintf(minute,    sizeof(minute), "%02d", ptm->tm_min);
    snprintf(second,    sizeof(second), "%02d", ptm->tm_sec);

    // ~ Get the host name to add inside the log name.
    char host[35] = "Host";
    int host_error = 0;
    host_error = gethostname(host, 35);

    if (host_error != 0) {
        fprintf(stderr, "[ERROR in gethostname(char *, int)] Could not access the host name\n");
        return 1;
    }

    strcpy(log_file_name_local, year);             strcat(log_file_name_local, "_");
    strcat(log_file_name_local, month);            strcat(log_file_name_local, "_");
    strcat(log_file_name_local, day);              strcat(log_file_name_local, "_");

    strcat(log_file_name_local, hour);             strcat(log_file_name_local, "_");
    strcat(log_file_name_local, minute);           strcat(log_file_name_local, "_");
    strcat(log_file_name_local, second);           strcat(log_file_name_local, "_");

    strcat(log_file_name_local, benchmark_name);   strcat(log_file_name_local, "_");
    strcat(log_file_name_local, host);
    strcat(log_file_name_local, ".log");

    absolute_path=getValueConfig_local(logdir_key);
    if(!absolute_path){
        fprintf(stderr, "[ERROR] Could not read log dir in config file '%s'\n",config_file_local);
	return 1;//exit(1);
    }
    if(!absolute_path){
        absolute_path = (char *)malloc(sizeof(char));
        absolute_path[0]='\0';
    }
    strcpy(full_log_file_name_local, absolute_path);
    if(strlen(absolute_path) > 0 && absolute_path[strlen(absolute_path)-1] != '/' )
        strcat(full_log_file_name_local, "/");
    strcat(full_log_file_name_local, log_file_name_local);
// ~ printf("%s\n", full_log_file_name_local);

    struct stat buf;
    if (stat(full_log_file_name_local, &buf) == 0) {
        fprintf(stderr, "[ERROR in create_log_file(char *)] File already exists %s\n",full_log_file_name_local);
        return 1;
    }

    FILE *file = NULL;

    file = fopen(full_log_file_name_local, "a");
    if (file == NULL){
        fprintf(stderr, "[ERROR in create_log_file(char *)] Unable to open file %s\n",full_log_file_name_local);
        return 1;
    }
    else if(test_info != NULL) {
        fprintf(file, "#HEADER %s\n",test_info);
    }
    else {
        fprintf(file, "#HEADER\n");
    }

    fprintf(file, "#BEGIN Y:%s M:%s D:%s Time:%s:%s:%s\n", year, month, day, hour, minute, second);
    fflush(file);
    fclose(file);

    kernels_total_errors_local = 0;
    iteration_number_local = 0;
    kernel_time_local_acc_local = 0;

    return 0;
};

// ~ ===========================================================================
// Log the string "#END" and reset global variables
int end_log_file_local(){
    FILE *file = NULL;

    file = fopen(full_log_file_name_local, "a");
    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name_local);
        return 1;
    }

    fprintf(file, "#END\n");
    fflush(file);
    fclose(file);
    kernels_total_errors_local = 0;
    iteration_number_local = 0;
    kernel_time_local_acc_local = 0;
    strcpy(log_file_name_local, "");
    strcpy(absolute_path, "");
    strcpy(full_log_file_name_local, "");

    return 0;
};

// ~ ===========================================================================
// Start time to measure kernel time, also update iteration number and log to file
int start_iteration_local(){

    update_timestamp_local();

/*
    FILE *file = fopen(full_log_file_name_local, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name_local);
        return 1;
    }

    fprintf(file, "#ITER it:%lu\n", iteration_number_local);
    fflush(file);
    fclose(file);
    iteration_number_local++;
*/
    log_error_detail_count=0;
    it_time_start_local = get_time();
    return 0;

};

// ~ ===========================================================================
// Finish the measured kernel time log both time (total time and kernel time)
int end_iteration_local(){

    update_timestamp_local();

    kernel_time_local = (double) (get_time() - it_time_start_local) / 1000000;
    kernel_time_local_acc_local += kernel_time_local;

    log_error_detail_count=0;

    if(iteration_number_local % iter_interval_print_local == 0) {

        FILE *file = fopen(full_log_file_name_local, "a");

        if (file == NULL){
            fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name_local);
            return 1;
        }

        fprintf(file,"#IT Ite:%lu KerTime:%f AccTime:%f\n", iteration_number_local, kernel_time_local, kernel_time_local_acc_local);
        //fprintf(file, "#TIME kernel_time_local:%f\n", kernel_time_local);
        //fprintf(file, "#ACC_TIME total_time:%f\n", kernel_time_local_acc_local);
        fflush(file);
        fclose(file);
    }

    iteration_number_local++;

    return 0;

};

// ~ ===========================================================================
// Update total errors variable and log both errors(total errors and kernel errors)
int log_error_count_local(unsigned long int kernel_errors){

    update_timestamp_local();

    if(kernel_errors < 1) {
        return 0;
    }

    kernels_total_errors_local += kernel_errors;

    FILE *file = NULL;
    file = fopen(full_log_file_name_local, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name_local);
        return 1;
    }

    // (iteration_number_local-1) because this function is called after end_iteration_local() that increments iteration_number_local
    fprintf(file, "#SDC Ite:%lu KerTime:%f AccTime:%f KerErr:%lu AccErr:%lu\n", iteration_number_local-1, kernel_time_local, kernel_time_local_acc_local, kernel_errors, kernels_total_errors_local);
    //fprintf(file, "#SDC kernel_errors:%lu\n", kernel_errors);
    //fprintf(file, "#TOTAL_SDC total_errors:%lu\n", kernels_total_errors_local);
    fflush(file);


    if(kernel_errors > max_errors_per_iter_local){
#ifdef ERR_INJ
        fprintf(file, "#ERR_INJ not aborting, we would abort otherwise\n");
#else
        fprintf(file, "#ABORT too many errors per iteration\n");
        fflush(file);
        fclose(file);
        end_log_file_local();
        exit(1);
#endif
    }


    if(kernel_errors == last_iter_errors_local && (last_iter_with_errors_local+1) == iteration_number_local && kernel_errors != 0){
        fprintf(file, "#ABORT amount of errors equals of the last iteration\n");
        fflush(file);
        fclose(file);
        end_log_file_local();
        exit(1);
    }

    fclose(file);

    last_iter_errors_local = kernel_errors;
    last_iter_with_errors_local = iteration_number_local;

    return 0;

};

// ~ ===========================================================================
// Print some string with the detail of an error to log file
int log_error_detail_local(char *string){
    FILE *file = NULL;

    #pragma omp parallel shared(log_error_detail_count) 
    {
        #pragma omp critical 
        log_error_detail_count++;
    }
    // Limits the number of lines written to logfile so that 
    // HD space will not explode
    if((unsigned long)log_error_detail_count > max_errors_per_iter_local)
        return 0;

    file = fopen(full_log_file_name_local, "a");
    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name_local);
        return 1;
    }

    fputs("#ERR ", file);
    fputs(string, file);
    fprintf(file, "\n");
    fflush(file);
    fclose(file);
    return 0;
};

//=======================================
//***************************************
//=======================================

// Location of timetamp file for software watchdog
//char timestamp_watchdog[200] = "/home/carol/watchdog/timestamp.txt";
char *timestamp_watchdog;
char timestamp_file[] = "timestamp.txt";
char config_file[]="/etc/radiation-benchmarks.conf";
char vardir_key[]="vardir";

// Max errors that can be found for a single iteration
// If more than max errors is found, exit the program
unsigned long int max_errors_per_iter = 500;

// Used to print the log only for some iterations, equal 1 means print every iteration
int iter_interval_print = 1;

char log_file_name[200] = "";
char full_log_file_name[500] = "";

// Saves the last amount of error found for a specific iteration
unsigned long int last_iter_errors = 0;
// Saves the last iteration index that had an error
unsigned long int last_iter_with_errors = 0;

unsigned long int kernels_total_errors = 0;
unsigned long int iteration_number = 0;
double kernel_time_acc = 0;
double kernel_time = 0;
long long it_time_start;

char message[2000];

//code to send signal to server that controls switch PING
#define PORT "8888" // the port client will be connecting to 
#define SERVER_IP "143.54.10.183" //"127.0.0.1"
// get sockaddr, IPv4 or IPv6:

void *get_in_addr(struct sockaddr *sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}

	return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

void send_message(char * message){
	int sockfd;
	struct addrinfo hints, *servinfo, *p;
	int rv;
	char s[INET6_ADDRSTRLEN];
	char full_message[2500]="";


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


	freeaddrinfo(servinfo); // all done with this structure

	strcpy(full_message, full_log_file_name);
	strcat(full_message, "|");
	strcat(full_message, message);
	if(send(sockfd , full_message, 2500 , 0)<0)
        {
            fprintf(stderr, "Send log message failed");
        }
	close(sockfd);
}

// ~ ===========================================================================
unsigned long int set_max_errors_iter(unsigned long int max_errors){
     set_max_errors_iter_local(max_errors);
     max_errors_per_iter = max_errors;

     return max_errors_per_iter;
};

// ~ ===========================================================================
// Set the interval the program must print log details, default is 1 (each iteration)
int set_iter_interval_print(int interval){
    set_iter_interval_print_local(interval);

    if(interval < 1) {
        iter_interval_print = 1;
    }
    else {
        iter_interval_print = interval;
    }

    return iter_interval_print;
};

// ~ ===========================================================================
// Update with current timestamp the file where the software watchdog watchs
void update_timestamp() {
    time_t timestamp = time(NULL);
    char time_s[50];
    char string[100] = "echo ";

    sprintf(time_s, "%d", (int) timestamp);

    strcat(string, time_s);
    strcat(string, " > ");
    strcat(string, timestamp_watchdog);
    system(string);
    int res = system(string);
};

// ~ ===========================================================================
// Read config file to get the value of a 'key = value' pair
char * getValueConfig(char * key){
	FILE * fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	char value[200];
	int i,j;
	int key_not_match;

	fp = fopen(config_file, "r");
	if (fp == NULL)
		return NULL;

	while ((read = getline(&line, &len, fp)) != -1) {
		// ignore comments and sections in config file
		if(line[0] == '#' || line[0] == '[')
			continue;

		// remove white spaces
		for(i=0;line[i]==' '; i++);
		// check if key of this line is the key we are looking for
		j=0;
		key_not_match=0;
		for(; line[i]!= ' ' && line[i] != '=' && key[j]!= '\0'; i++){
			if(key[j]!=line[i]){
				key_not_match=1;
				break;
			}
			j++;
		}
		// Key not matched
		if(key_not_match)
			continue;
		// key of line is a substring of the key we are looking for
		if(key[j]!='\0')
			continue;
		// key matched but is a substring of current key
		if(line[i] !=' ' && line[i]!='=')
			continue;
		// ignore spaces and '=' to go the the frist character of value
		for(;line[i]==' '||line[i] == '='; i++);
		j=0;
		// copy value to buffer until end of line or '#' is found
		for(;line[i]!='\0'&&line[i]!='#'&&line[i]!='\n';i++){
			value[j]=line[i];
			j++;
		}
		value[j]='\0';
		char *v = (char *)malloc(sizeof(char)*strlen(value)+2);
		strcpy(v, value);
		fclose(fp);
		if (line)
			free(line);
		return v;
	}

	fclose(fp);
	if (line)
		free(line);
	return NULL;
};

// ~ ===========================================================================
// Return the name of the log file generated
char * get_log_file_name(){
    return full_log_file_name;
};

// ~ ===========================================================================
// Generate the log file name, log info from user about the test to be executed and reset log variables
int start_log_file(char *benchmark_name, char *test_info){
    start_log_file_local(benchmark_name, test_info);

    char *var_dir=getValueConfig(vardir_key);
    if(!var_dir){
        fprintf(stderr, "[ERROR] Could not read var dir in config file '%s'\n",config_file);
	return 1;//exit(1);
    }
    timestamp_watchdog = (char *)malloc(sizeof(char)* (strlen(var_dir)+strlen(timestamp_file)+4) );
    strcpy(timestamp_watchdog, var_dir);
    if(strlen(timestamp_watchdog) > 0 && timestamp_watchdog[strlen(timestamp_watchdog)-1] != '/' )
        strcat(timestamp_watchdog, "/");
    strcat(timestamp_watchdog, timestamp_file);

    update_timestamp();

    time_t file_time;
    struct tm *ptm;
    char day[10], month[10], year[15], hour[10], second[10], minute[10];
    //char log_file_name[180] = "";

    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day,       sizeof(day),    "%02d", ptm->tm_mday);
    snprintf(month,     sizeof(month),  "%02d", ptm->tm_mon+1);
    snprintf(year,      sizeof(year),   "%04d", ptm->tm_year+1900);
    snprintf(hour,      sizeof(hour),   "%02d", ptm->tm_hour);
    snprintf(minute,    sizeof(minute), "%02d", ptm->tm_min);
    snprintf(second,    sizeof(second), "%02d", ptm->tm_sec);

    // ~ Get the host name to add inside the log name.
    char host[35] = "Host";
    int host_error = 0;
    host_error = gethostname(host, 35);

    if (host_error != 0) {
        fprintf(stderr, "[ERROR in gethostname(char *, int)] Could not access the host name\n");
        return 1;
    }

    strcpy(log_file_name, year);             strcat(log_file_name, "_");
    strcat(log_file_name, month);            strcat(log_file_name, "_");
    strcat(log_file_name, day);              strcat(log_file_name, "_");

    strcat(log_file_name, hour);             strcat(log_file_name, "_");
    strcat(log_file_name, minute);           strcat(log_file_name, "_");
    strcat(log_file_name, second);           strcat(log_file_name, "_");

    strcat(log_file_name, benchmark_name);   strcat(log_file_name, "_");
    strcat(log_file_name, host);
    strcat(log_file_name, ".log");


    strcpy(full_log_file_name, log_file_name);
    if(test_info != NULL) {
        snprintf(message, sizeof(message),"#HEADER %s\n", test_info);
	send_message(message);
    }
    else {
	send_message("#HEADER\n");
    }

    snprintf(message, sizeof(message),"#BEGIN Y:%s M:%s D:%s Time:%s:%s:%s\n", year, month, day, hour, minute, second);
    send_message(message);

    kernels_total_errors = 0;
    iteration_number = 0;
    kernel_time_acc = 0;

    return 0;
};

// ~ ===========================================================================
// Log the string "#END" and reset global variables
int end_log_file(){
    end_log_file_local();

    send_message("#END");
    kernels_total_errors = 0;
    iteration_number = 0;
    kernel_time_acc = 0;
    strcpy(log_file_name, "");
    strcpy(full_log_file_name, "");

    return 0;
};

// ~ ===========================================================================
// Start time to measure kernel time, also update iteration number and log to file
int start_iteration(){
    update_timestamp();

    it_time_start = get_time();
    start_iteration_local();
    return 0;

};

// ~ ===========================================================================
// Finish the measured kernel time log both time (total time and kernel time)
int end_iteration(){

    update_timestamp();

    kernel_time = (double) (get_time() - it_time_start) / 1000000;
    kernel_time_acc += kernel_time;



    if(iteration_number % iter_interval_print == 0) {
        snprintf(message, sizeof(message),"#IT Ite:%lu KerTime:%f AccTime:%f\n", iteration_number, kernel_time, kernel_time_acc);
	send_message(message);
    }

    iteration_number++;
    end_iteration_local();
    return 0;

};

// ~ ===========================================================================
// Update total errors variable and log both errors(total errors and kernel errors)
int log_error_count(unsigned long int kernel_errors){

    update_timestamp();

    if(kernel_errors < 1) {
        return 0;
    }

    kernels_total_errors += kernel_errors;


    // (iteration_number-1) because this function is called after end_iteration() that increments iteration_number
    snprintf(message, sizeof(message),"#SDC Ite:%lu KerTime:%f AccTime:%f KerErr:%lu AccErr:%lu\n", iteration_number-1, kernel_time, kernel_time_acc, kernel_errors, kernels_total_errors);
    send_message(message);


    if(kernel_errors > max_errors_per_iter){
	send_message("#ABORT too many errors per iteration\n");
        end_log_file();
        end_log_file_local();
        exit(1);
    }


    if(kernel_errors == last_iter_errors && (last_iter_with_errors+1) == iteration_number && kernel_errors != 0){
	send_message("#ABORT amount of errors equals of the last iteration\n");
        end_log_file();
        end_log_file_local();
        exit(1);
    }

    last_iter_errors = kernel_errors;
    last_iter_with_errors = iteration_number;
    log_error_count_local(kernel_errors);
    return 0;

};

// ~ ===========================================================================
// Print some string with the detail of an error to log file
int log_error_detail(char *string){

    snprintf(message, sizeof(message),"#ERR %s\n", string);
    send_message(message);
    log_error_detail_local(string);
    return 0;

};

