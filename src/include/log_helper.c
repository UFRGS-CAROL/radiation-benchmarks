#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

//Buff for ECC check
#define BUFSIZE 128

#include "log_helper.h"

#ifdef MIC_NATIVE

char timestamp_watchdog[200] = "/micNfs/carol/logs/timestamp.txt";
#define QUERY_GPU "echo Enabled"
#define ENABLED_CONFIRMATION "Enabled"

#else

// Location of timetamp file for software watchdog
//char timestamp_watchdog[200] = "/home/carol/watchdog/timestamp.txt";
char *timestamp_watchdog;
char timestamp_file[] = "timestamp.txt";
char vardir_key[] = "vardir";

//Terminal query which will tells if ECC is enable or not, it could vary depend the platform
#define QUERY_GPU "/usr/bin/nvidia-smi --query-gpu=gpu_name,ecc.mode.current --format=csv,noheader 2>/tmp/trash"
#define ENABLED_CONFIRMATION "Enabled"

#endif

// Max errors that can be found for a single iteration
// If more than max errors is found, exit the program
unsigned long int max_errors_per_iter = 500;
unsigned long int max_infos_per_iter = 500;

// Absolute path for log file, if needed
#ifdef MIC_NATIVE

char absolute_path[200] = "/micNfs/carol/logs/";

#else

//char absolute_path[200] = "/home/carol/logs/";
char *absolute_path;

#endif

char logdir_key[] = "logdir";
char signalcmd_key[] = "signalcmd";
#ifdef MIC_NATIVE
char config_file[] = "/micNfs/radiation-benchmarks.conf";
#else
char config_file[] = "/etc/radiation-benchmarks.conf";
#endif

// Used to print the log only for some iterations, equal 1 means print every iteration
int iter_interval_print = 1;

// Used to log max_error_per_iter details each iteration
int log_error_detail_count = 0;
int log_info_detail_count = 0;

char log_file_name[200] = "";
char full_log_file_name[300] = "";

// Saves the last amount of error found for a specific iteration
unsigned long int last_iter_errors = 0;
// Saves the last iteration index that had an error
unsigned long int last_iter_with_errors = 0;

unsigned long int kernels_total_errors = 0;
unsigned long int kernels_total_infos = 0;
unsigned long int iteration_number = 0;
double kernel_time_acc = 0;
double kernel_time = 0;
long long it_time_start;

// ~ ===========================================================================
// Functions to check ECC
/**
 * String contains
 * check if word contains in sent string
 * return 1 if contains
 * return 0 otherwise
 */
int contains(const char *sent, char *word) {
    //call popen on terminal---------------
    const char *temp = strstr(sent, word);
    if (temp) {
        return 1;
    }

    return 0;
}

// ~ ===========================================================================
/**
 * popen_call
 * call popen and check if check_line is in output string
 * if check_line is in popen output an output is writen in output_line
 * return 1 if the procedure executed
 * return 0 otherwise
 */
int popen_call(char *cmd, char *check_line, char *output_line) {
    FILE *fp;
    char buf[BUFSIZE];
    int ret = 0;
    if ((fp = popen(cmd, "r")) == NULL) {
        //printf("Error opening pipe!\n");
        return 0;
    }

    while (fgets(buf, BUFSIZE, fp) != NULL) {
        if (contains(buf, check_line)) {
            strcpy(output_line, buf);
            ret = 1;
        }
    }


    fflush(fp);
    if (pclose(fp)) {
        //printf("Command not found or exited with error status\n");
        return 0;
    }
    return ret;
}

// ~ ===========================================================================
/**
 * This functions checks if ECC is enable or disabled for NVIDIA GPUs
 * 0 if ECC is disabled
 * 1 if ECC is enabled
 */
int check_ecc_status() {
    char output_line[BUFSIZE];
    memset(output_line, 0, BUFSIZE);

    //check for enabled ECC
    return (popen_call(QUERY_GPU, ENABLED_CONFIRMATION, output_line));
}

// ~ ===========================================================================
long long get_time() {
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// ~ ===========================================================================
unsigned long int set_max_errors_iter(unsigned long int max_errors) {
    max_errors_per_iter = max_errors;

    return max_errors_per_iter;
}

// ~ ===========================================================================
unsigned long int set_max_infos_iter(unsigned long int max_infos) {
    max_infos_per_iter = max_infos;

    return max_infos_per_iter;
}

// ~ ===========================================================================
// Set the interval the program must print log details, default is 1 (each iteration)
int set_iter_interval_print(int interval) {
    if (interval < 1) {
        iter_interval_print = 1;
    } else {
        iter_interval_print = interval;
    }

    return iter_interval_print;
}

// ~ ===========================================================================
// Read config file to get the value of a 'key = value' pair
char * getValueConfig(char * key) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char value[200];
    int i, j;
    int key_not_match;

    fp = fopen(config_file, "r");
    if (fp == NULL)
        return NULL;

    while ((read = getline(&line, &len, fp)) != -1) {
        // ignore comments and sections in config file
        if (line[0] == '#' || line[0] == '[')
            continue;

        // remove white spaces
        for (i = 0; line[i] == ' '; i++)
            ;
        // check if key of this line is the key we are looking for
        j = 0;
        key_not_match = 0;
        for (; line[i] != ' ' && line[i] != '=' && key[j] != '\0'; i++) {
            if (key[j] != line[i]) {
                key_not_match = 1;
                break;
            }
            j++;
        }
        // Key not matched
        if (key_not_match)
            continue;
        // key of line is a substring of the key we are looking for
        if (key[j] != '\0')
            continue;
        // key matched but is a substring of current key
        if (line[i] != ' ' && line[i] != '=')
            continue;
        // ignore spaces and '=' to go the the frist character of value
        for (; line[i] == ' ' || line[i] == '='; i++)
            ;
        j = 0;
        // copy value to buffer until end of line or '#' is found
        for (; line[i] != '\0' && line[i] != '#' && line[i] != '\n'; i++) {
            value[j] = line[i];
            j++;
        }
        value[j] = '\0';
        char *v = (char *) malloc(sizeof(char) * strlen(value) + 2);
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
}

// ~ ===========================================================================
// Update with current timestamp the file where the software watchdog watchs
void update_timestamp() {
    char *signalcmd = getValueConfig(signalcmd_key);
    system(signalcmd);
    time_t timestamp = time(NULL);
    FILE *fp = fopen(timestamp_watchdog, "w");
    if (fp) {
        fprintf(fp, "%d", (int) timestamp);
        fclose(fp);
    }
}

// ~ ===========================================================================
// Return the name of the log file generated
char * get_log_file_name() {
    return full_log_file_name;
}

// ~ ===========================================================================
// Generate the log file name, log info from user about the test to be executed and reset log variables
int start_log_file(char *benchmark_name, char *test_info) {

#ifndef MIC_NATIVE
    char *var_dir = getValueConfig(vardir_key);
    if (!var_dir) {
        fprintf(stderr, "[ERROR] Could not read var dir in config file '%s'\n",
                config_file);
        return 1; //exit(1);
    }
    timestamp_watchdog = (char *) malloc(
            sizeof(char) * (strlen(var_dir) + strlen(timestamp_file) + 4));
    strcpy(timestamp_watchdog, var_dir);
    if (strlen(timestamp_watchdog) > 0
            && timestamp_watchdog[strlen(timestamp_watchdog) - 1] != '/')
        strcat(timestamp_watchdog, "/");
    strcat(timestamp_watchdog, timestamp_file);
#endif
    update_timestamp();

    time_t file_time;
    struct tm *ptm;
    char day[10], month[10], year[15], hour[10], second[10], minute[10];
    char log_file_name[190] = "";

    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day, sizeof(day), "%02d", ptm->tm_mday);
    snprintf(month, sizeof(month), "%02d", ptm->tm_mon + 1);
    snprintf(year, sizeof(year), "%04d", ptm->tm_year + 1900);
    snprintf(hour, sizeof(hour), "%02d", ptm->tm_hour);
    snprintf(minute, sizeof(minute), "%02d", ptm->tm_min);
    snprintf(second, sizeof(second), "%02d", ptm->tm_sec);

    // ~ Get the host name to add inside the log name.
    char host[35] = "Host";
    int host_error = 0;
    host_error = gethostname(host, 35);

    if (host_error != 0) {
        fprintf(stderr,
                "[ERROR in gethostname(char *, int)] Could not access the host name\n");
        return 1;
    }

    strcpy(log_file_name, year);
    strcat(log_file_name, "_");
    strcat(log_file_name, month);
    strcat(log_file_name, "_");
    strcat(log_file_name, day);
    strcat(log_file_name, "_");

    strcat(log_file_name, hour);
    strcat(log_file_name, "_");
    strcat(log_file_name, minute);
    strcat(log_file_name, "_");
    strcat(log_file_name, second);
    strcat(log_file_name, "_");

    strcat(log_file_name, benchmark_name);
    strcat(log_file_name, "_");
    //check ECC
    if(check_ecc_status()){
        strcat(log_file_name, "ECC_ON_");
    }else{
        strcat(log_file_name, "ECC_OFF_");
    }
    //--------
    strcat(log_file_name, host);
    strcat(log_file_name, ".log");

#ifndef MIC_NATIVE
    absolute_path = getValueConfig(logdir_key);
    if (!absolute_path) {
        fprintf(stderr, "[ERROR] Could not read log dir in config file '%s'\n",
                config_file);
        return 1; //exit(1);
    }
    if (!absolute_path) {
        absolute_path = (char *) malloc(sizeof(char));
        absolute_path[0] = '\0';
    }
#endif
    strcpy(full_log_file_name, absolute_path);
    if (strlen(absolute_path) > 0
            && absolute_path[strlen(absolute_path) - 1] != '/')
        strcat(full_log_file_name, "/");
    strcat(full_log_file_name, log_file_name);
// ~ printf("%s\n", full_log_file_name);

    struct stat buf;
    if (stat(full_log_file_name, &buf) == 0) {
        fprintf(stderr,
                "[ERROR in create_log_file(char *)] File already exists %s\n",
                full_log_file_name);
        return 1;
    }

    FILE *file = NULL;

    file = fopen(full_log_file_name, "a");
    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in create_log_file(char *)] Unable to open file %s\n",
                full_log_file_name);
        return 1;
    } else if (test_info != NULL) {
        fprintf(file, "#HEADER %s\n", test_info);
    } else {
        fprintf(file, "#HEADER\n");
    }

    fprintf(file, "#BEGIN Y:%s M:%s D:%s Time:%s:%s:%s\n", year, month, day,
            hour, minute, second);
    fflush(file);
    fclose(file);

    kernels_total_errors = 0;
    iteration_number = 0;
    kernel_time_acc = 0;

    return 0;
}

// ~ ===========================================================================
// Log the string "#END" and reset global variables
int end_log_file() {
    FILE *file = NULL;

    file = fopen(full_log_file_name, "a");
    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in log_string(char *)] Unable to open file %s\n",
                full_log_file_name);
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

// ~ ===========================================================================
// Start time to measure kernel time, also update iteration number and log to file
int start_iteration() {

    update_timestamp();

    /*
     FILE *file = fopen(full_log_file_name, "a");

     if (file == NULL){
     fprintf(stderr, "[ERROR in log_string(char *)] Unable to open file %s\n",full_log_file_name);
     return 1;
     }

     fprintf(file, "#ITER it:%lu\n", iteration_number);
     fflush(file);
     fclose(file);
     iteration_number++;
     */
    log_error_detail_count = 0;
    log_info_detail_count = 0;
    it_time_start = get_time();
    return 0;

}

// ~ ===========================================================================
// Finish the measured kernel time log both time (total time and kernel time)
int end_iteration() {

    update_timestamp();

    kernel_time = (double) (get_time() - it_time_start) / 1000000;
    kernel_time_acc += kernel_time;

    log_error_detail_count = 0;
    log_info_detail_count = 0;

    if (iteration_number % iter_interval_print == 0) {

        FILE *file = fopen(full_log_file_name, "a");

        if (file == NULL) {
            fprintf(stderr,
                    "[ERROR in log_string(char *)] Unable to open file %s\n",
                    full_log_file_name);
            return 1;
        }

        fprintf(file, "#IT Ite:%lu KerTime:%f AccTime:%f\n", iteration_number,
                kernel_time, kernel_time_acc);
        //fprintf(file, "#TIME kernel_time:%f\n", kernel_time);
        //fprintf(file, "#ACC_TIME total_time:%f\n", kernel_time_acc);
        fflush(file);
        fclose(file);
    }

    iteration_number++;

    return 0;

}

// ~ ===========================================================================
// Update total errors variable and log both errors(total errors and kernel errors)
int log_error_count(unsigned long int kernel_errors) {

    update_timestamp();

    if (kernel_errors < 1) {
        return 0;
    }

    kernels_total_errors += kernel_errors;

    FILE *file = NULL;
    file = fopen(full_log_file_name, "a");

    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in log_string(char *)] Unable to open file %s\n",
                full_log_file_name);
        return 1;
    }

    // (iteration_number-1) because this function is called after end_iteration() that increments iteration_number
    fprintf(file, "#SDC Ite:%lu KerTime:%f AccTime:%f KerErr:%lu AccErr:%lu\n",
            iteration_number - 1, kernel_time, kernel_time_acc, kernel_errors,
            kernels_total_errors);
    //fprintf(file, "#SDC kernel_errors:%lu\n", kernel_errors);
    //fprintf(file, "#TOTAL_SDC total_errors:%lu\n", kernels_total_errors);
    fflush(file);

    if (kernel_errors > max_errors_per_iter) {
#ifdef ERR_INJ
        fprintf(file, "#ERR_INJ not aborting, we would abort otherwise\n");
#else
        fprintf(file, "#ABORT too many errors per iteration\n");
        fflush(file);
        fclose(file);
        end_log_file();
        exit(1);
#endif
    }

    if (kernel_errors == last_iter_errors
            && (last_iter_with_errors + 1) == iteration_number
            && kernel_errors != 0) {
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

// ~ ===========================================================================
// Update total infos variable and log both infos(total infos and iteration infos)
int log_info_count(unsigned long int info_count) {

    update_timestamp();

    if (info_count < 1) {
        return 0;
    }

    kernels_total_infos += info_count;

    FILE *file = NULL;
    file = fopen(full_log_file_name, "a");

    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in log_string(char *)] Unable to open file %s\n",
                full_log_file_name);
        return 1;
    }

    // (iteration_number-1) because this function is called after end_iteration() that increments iteration_number
    fprintf(file, "#CINF Ite:%lu KerTime:%f AccTime:%f KerInfo:%lu AccInfo:%lu\n",
            iteration_number - 1, kernel_time, kernel_time_acc, info_count,
            kernels_total_infos);
    //fprintf(file, "#SDC kernel_errors:%lu\n", kernel_errors);
    //fprintf(file, "#TOTAL_SDC total_errors:%lu\n", kernels_total_errors);
    fflush(file);

//     if (info_count > max_infos_per_iter) {
// #ifdef ERR_INJ
//         fprintf(file, "#ERR_INJ not aborting, we would abort otherwise\n");
// #else
//         fprintf(file, "#ABORT too many infos per iteration\n");
//         fflush(file);
//         fclose(file);
//         end_log_file();
//         exit(1);
// #endif
//     }

    // if (kernel_errors == last_iter_errors
    //         && (last_iter_with_errors + 1) == iteration_number
    //         && kernel_errors != 0) {
    //     fprintf(file, "#ABORT amount of errors equals of the last iteration\n");
    //     fflush(file);
    //     fclose(file);
    //     end_log_file();
    //     exit(1);
    // }

    fclose(file);

    // last_iter_errors = kernel_errors;
    // last_iter_with_errors = iteration_number;

    return 0;

}

// ~ ===========================================================================
// Print some string with the detail of an error to log file
int log_error_detail(char *string) {
    FILE *file = NULL;

#pragma omp parallel shared(log_error_detail_count)
    {
#pragma omp critical
        log_error_detail_count++;
    }
    // Limits the number of lines written to logfile so that
    // HD space will not explode
    if ((unsigned long) log_error_detail_count > max_errors_per_iter)
        return 0;

    file = fopen(full_log_file_name, "a");
    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in log_string(char *)] Unable to open file %s\n",
                full_log_file_name);
        return 1;
    }

    fputs("#ERR ", file);
    fputs(string, file);
    fprintf(file, "\n");
    fflush(file);
    fclose(file);
    return 0;
}

// ~ ===========================================================================
// Print some string with the detail of an error/information to log file
int log_info_detail(char *string) {
    FILE *file = NULL;

#pragma omp parallel shared(log_info_detail_count)
    {
#pragma omp critical
        log_info_detail_count++;
    }
    // Limits the number of lines written to logfile so that
    // HD space will not explode
    if ((unsigned long) log_info_detail_count > max_infos_per_iter)
        return 0;

    file = fopen(full_log_file_name, "a");
    if (file == NULL) {
        fprintf(stderr,
                "[ERROR in log_string(char *)] Unable to open file %s\n",
                full_log_file_name);
        return 1;
    }

    fputs("#INF ", file);
    fputs(string, file);
    fprintf(file, "\n");
    fflush(file);
    fclose(file);
    return 0;
}

unsigned long int get_iteration_number()
{
    return iteration_number;
}
