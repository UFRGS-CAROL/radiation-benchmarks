#include <stdio.h>
#include <string.h>
#include <time.h>

char log_file_name[100] = "";
char absolute_path[200] = "";
char full_log_file_name[300] = "";

double kernels_total_time = 0;
unsigned long int kernels_total_errors = 0;
unsigned long int iteration_number = 0;

// In case the user needs the log to be generated in some exact absolute path
void set_absolute_path(char *path){
	strcpy(absolute_path, path);
}

// Return the name of the log file generated
char * get_log_file_name(){
	return full_log_file_name;
}

// Generate the log file name and reset log variables
int start_log_file(char *benchmark_name){

	time_t file_time;
	struct tm *ptm;
	char day[2], month[2], year[4], hour[2], second[2], minute[2];
	char log_file_name[60];

	file_time = time(NULL);
	ptm = gmtime(&file_time);

	snprintf(day, sizeof(day + 1), "%d", ptm->tm_mday);
	snprintf(month, sizeof(month + 1), "%d", ptm->tm_mon+1);
	snprintf(year, sizeof(year + 1), "%d", ptm->tm_year+1900);
	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	strcpy(log_file_name,day);strcat(log_file_name,"_");
	strcat(log_file_name,month);strcat(log_file_name,"_");
	strcat(log_file_name,year);strcat(log_file_name,"_");
	strcat(log_file_name,hour);strcat(log_file_name,"_");
	strcat(log_file_name,minute);strcat(log_file_name,"_");
	strcat(log_file_name,second);strcat(log_file_name,"_");
	strcat(log_file_name,benchmark_name);
	strcat(log_file_name,".log");

	strcpy(full_log_file_name, absolute_path);
	if(strlen(absolute_path) > 0 && absolute_path[strlen(absolute_path)-1] != '/' )
		strcat(full_log_file_name, "/");
	strcat(full_log_file_name, log_file_name);

	FILE *file = fopen(full_log_file_name, "a");
	if (file == NULL){
		fprintf(stderr, "[ERROR in create_log_file(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	} else
		close(file);


	kernels_total_time = 0;
	kernels_total_errors = 0;
	iteration_number = 0;
	return 0;
}


// Print some string to log file
int log_string(char *string){

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fputs(string, file);
	fprintf(file, "\n");
	fflush(file);
	close(file);
	return 0;
}


// Print some string to log file followed by some integer value
int log_string_int(char *string, int value){

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fputs(string, file);
	fprintf(file, "%d\n", value);
	fflush(file);
	close(file);
	return 0;
}


// Print some string to log file followed by some double value
int log_string_double(char *string, double value){

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fputs(string, file);
	fprintf(file, "%f\n", value);
	fflush(file);
	close(file);
	return 0;
}

// Update total time variable and log both time (total time and kernel time)
int log_time(double kernel_time){

	kernels_total_time += kernel_time;

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fprintf(file, "kernel time: %f\n", kernel_time);
	fprintf(file, "kernels total time: %f\n", kernels_total_time);
	fflush(file);
	close(file);
	return 0;

}

// Update total errors variable and log both errors(total errors and kernel errors)
int log_error(unsigned long int kernel_errors){

	kernels_total_errors += kernel_errors;

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fprintf(file, "kernel errors: %lu\n", kernel_errors);
	fprintf(file, "kernels total errors: %lu\n", kernels_total_errors);
	fflush(file);
	close(file);
	return 0;

}

// Update total errors variable and log both errors(total errors and kernel errors)
int start_iteration(){

	FILE *file = fopen(full_log_file_name, "a");

	if (file == NULL){
		fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
		return 1;
	}

	fprintf(file, "---it: %lu\n", iteration_number);
	fflush(file);
	close(file);
	return 0;

}
