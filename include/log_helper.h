
// Optional, in case user want to specify tha absolute path 
// where the log must be generated
void set_absolute_path(char *path);

// Return the name of the log file generated
char * get_log_file_name();

// Generate the log file name, log info from user about the test
// to be executed and reset log variables
int start_log_file(char *benchmark_name, char *test_info);

// Log the string "#END" and reset global variables
int end_log_file();

// Start time to measure kernel time, also update 
// iteration number and log to file
int start_iteration();

// Finish the measured kernel time log both 
// time (total time and kernel time)
int end_iteration();

// Update total errors variable and log both 
// errors(total errors and kernel errors)
int log_error_count(unsigned long int kernel_errors);

// Print some string with the detail of an error to log file
int log_error_detail(char *string);
