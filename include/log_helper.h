
// Optional, in case user want to specify tha absolute path 
// where the log must be generated
void set_absolute_path(char *path);

// Generate a new log file and reset any control variable
int start_log_file(char *benchmark_name);

// Return the name of the log file generated
char * get_log_file_name();

// Print a string with an incremental number of iteration
int start_iteration();

// Print time and the total time inserted since the call 
// of start_log_file(char *) function
int log_time(double kernel_time);

// Print error and the total errors inserted since the call 
// of start_log_file(char *) function
int log_error(unsigned long int kernel_errors);

// Print some string to log file
int log_string(char *string);

// Print some string to log file followed by an int value, DO NOT insert 
// string formatter such as %d and %f
// i.e. log_string_int("value: ", 5); produces line: "value: 5"
int log_string_int(char *string, int value);

// Print some string to log file followed by a double value, DO NOT insert 
// string formatter such as %d and %f
// i.e. log_string_double("value: ", 5.0); produces line: "value: 5.000000"
int log_string_doulbe(char *string, double value);
