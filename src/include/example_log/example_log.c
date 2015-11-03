#include <stdio.h>

#include "../log_helper.h"

int main(){

	// Start the log with filename including "my_benchmark" to its name, and
	// header inside log file will print the detail "size:x repetition:y"
	if(start_log_file((char*)"my_benchmark", (char*)"size:x repetition:y")){
		fprintf(stderr,"Could not start log file");
		exit(1);
	}
	// set the maximum number of errors allowed for each iteration,
	// default is 500
	set_max_errors_iter(32);

	// set the interval of iteration to print details of current test, 
	// default is 1
	set_iter_interval_print(5);
	
	printf("log file is %s\n",get_log_file_name());
	
	int i;
	for(i = 0; i < 40; i++){

		start_iteration();
		// Execute the test (ONLY THE KERNEL), log functions will measure kernel time
		sleep(1);
		end_iteration();


		int error_count = 0;
		
		if(i%8==0){
			// Testing with error_count > 0 for some iterations
			// You can call as many log_error_detail(str) as you need
			// However, it will log only the 500 errors or the
			// max_errors_iter set with set_max_errors_iter()
			log_error_detail("detail of error x");

			// Tell log how many errors the iteration had
			error_count = i+1;
		}

		// log how many errors the iteration had
		// if error_count is greater than 500, or the
		// max_errors_iter set with set_max_errors_iter()
		// it will terminate the execution
		log_error_count(error_count);
	}

	// Finish the log file
	end_log_file();
    
}
