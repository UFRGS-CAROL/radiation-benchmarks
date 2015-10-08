#include <stdio.h>

#include "../log_helper.h"

int main(){

	// Optional, default is /home/carol/logs/
	set_absolute_path("");
	start_log_file("my_benchmark", "size:x repetition:y");
	set_max_errors_iter(32);
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
			log_error_detail("detail of error x");

			// Tell log how many errors the iteration had
			error_count = i+1;
		}

		// log how many errors the iteration had
		log_error_count(error_count);
	}

	end_log_file();
    
}
