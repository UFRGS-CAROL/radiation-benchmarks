#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Location of timetamp file for software watchdog
char timestamp_watchdog[200] = "/home/carol/watchdog/timestamp.txt";
// Max errors that can be found for a single iteration
// If more than max errors is found, exit the program
unsigned long int max_errors_per_iter = 5000;

// Absolute path for log file, if needed
char absolute_path[200] = "";


char log_file_name[100] = "";
char full_log_file_name[300] = "";

// Saves the last amount of error found for a specific iteration
unsigned long int last_iter_errors = 0;
// Saves the last iteration index that had an error
unsigned long int last_iter_with_errors = 0;

unsigned long int kernels_total_errors = 0;
unsigned long int iteration_number = 0;
double kernel_time_acc = 0;
long long it_time_start;


inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

unsigned long int set_max_errors_iter(unsigned long int max_errors){
     max_errors_per_iter = max_errors;
     return max_errors_per_iter;
}

// Update with current timestamp the file where the software watchdog watchs
void update_timestamp() {
    time_t timestamp = time(NULL);
    char time_s[50];
    sprintf(time_s, "%d", (int) timestamp);
    char string[100] = "echo ";
    strcat(string, time_s);
    strcat(string, " > ");
    strcat(string, timestamp_watchdog);
    system(string);
}

// In case the user needs the log to be generated in some exact absolute path
void set_absolute_path(char *path){
    strcpy(absolute_path, path);
}

// Return the name of the log file generated
char * get_log_file_name(){
    return full_log_file_name;
}

// Generate the log file name, log info from user about the test to be executed and reset log variables
int start_log_file(char *benchmark_name, char *test_info){

    update_timestamp();

    time_t file_time;
    struct tm *ptm;
    char day[3], month[3], year[5], hour[3], second[3], minute[3];
    char log_file_name[80];

    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day,       sizeof(day + 1),    "%02d", ptm->tm_mday);
    snprintf(month,     sizeof(month + 1),  "%02d", ptm->tm_mon+1);
    snprintf(year,      sizeof(year + 1),   "%04d", ptm->tm_year+1900);
    snprintf(hour,      sizeof(hour + 1),   "%02d", ptm->tm_hour);
    snprintf(minute,    sizeof(minute + 1), "%02d", ptm->tm_min);
    snprintf(second,    sizeof(second + 1), "%02d", ptm->tm_sec);
    strcpy(log_file_name,day);      strcat(log_file_name,"_");
    strcat(log_file_name,month);    strcat(log_file_name,"_");
    strcat(log_file_name,year);     strcat(log_file_name,"_");
    strcat(log_file_name,hour);     strcat(log_file_name,"_");
    strcat(log_file_name,minute);   strcat(log_file_name,"_");
    strcat(log_file_name,second);   strcat(log_file_name,"_");
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
    if(test_info != NULL)
        fprintf(file, "#HEADER %s\n",test_info);
    else
        fprintf(file, "#HEADER\n");
    fprintf(file, "#BEGIN Year:%s Month:%s Day:%s Time:%s:%s:%s\n", year, month, day, hour, minute, second);
    fflush(file);
    fclose(file);


    kernels_total_errors = 0;
    iteration_number = 0;
    kernel_time_acc = 0;
    return 0;
}

// Log the string "#END" and reset global variables
int end_log_file(){
    FILE *file = fopen(full_log_file_name, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
        return 1;
    }

    fprintf(file, "#END");
    fflush(file);
    fclose(file);
    kernels_total_errors = 0;
    iteration_number = 0;
    kernel_time_acc = 0;
    strcpy(log_file_name, "");
    strcpy(absolute_path, "");
    strcpy(full_log_file_name, "");

    return 0;
}

// Start time to measure kernel time, also update iteration number and log to file
int start_iteration(){

    update_timestamp();

    FILE *file = fopen(full_log_file_name, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
        return 1;
    }

    fprintf(file, "#ITER it:%lu\n", iteration_number);
    fflush(file);
    fclose(file);
    iteration_number++;
    it_time_start = get_time();
    return 0;

}

// Finish the measured kernel time log both time (total time and kernel time)
int end_iteration(){

    update_timestamp();

    double kernel_time = (double) (get_time() - it_time_start) / 1000000;
    kernel_time_acc += kernel_time;

    FILE *file = fopen(full_log_file_name, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
        return 1;
    }

    fprintf(file, "#TIME kernel_time:%f\n", kernel_time);
    fprintf(file, "#ACC_TIME total_time:%f\n", kernel_time_acc);
    fflush(file);
    fclose(file);
    return 0;

}

// Update total errors variable and log both errors(total errors and kernel errors)
int log_error_count(unsigned long int kernel_errors){

    update_timestamp();

    kernels_total_errors += kernel_errors;

    FILE *file = fopen(full_log_file_name, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
        return 1;
    }

    fprintf(file, "#SDC kernel_errors:%lu\n", kernel_errors);
    fprintf(file, "#TOTAL_SDC total_errors:%lu\n", kernels_total_errors);
    fflush(file);


    if(kernel_errors > max_errors_per_iter){
        fprintf(file, "#ABORT too many errors per iteration\n");
        fflush(file);
        fclose(file);
        end_log_file();
        exit(1);
    }


    if(kernel_errors == last_iter_errors && (last_iter_with_errors+1) == iteration_number && kernel_errors != 0){
        fprintf(file, "#ABORT amount of errors equals of the last iteration\n");
        fflush(file);
        fclose(file);
        end_log_file();
        exit(1);
    }

    fclose(file);

    last_iter_errors = kernel_errors;
    last_iter_with_errors = iteration_number;

    return 0;

}

// Print some string with the detail of an error to log file
int log_error_detail(char *string){

    FILE *file = fopen(full_log_file_name, "a");

    if (file == NULL){
        fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
        return 1;
    }

    fputs("#ERROR ", file);
    fputs(string, file);
    fprintf(file, "\n");
    fflush(file);
    fclose(file);
    return 0;
}

