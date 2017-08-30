 /*log_helper.i */
%module log_helper
%inline %{
extern unsigned long int set_max_errors_iter(unsigned long int max_errors);
extern int set_iter_interval_print(int interval);
extern void update_timestamp();
extern char * get_log_file_name();
extern int start_log_file(char *benchmark_name, char *test_info);
extern int end_log_file();
extern int start_iteration();
extern int end_iteration();
extern int log_error_count(unsigned long int kernel_errors);
extern int log_error_detail(char *string);

%}
