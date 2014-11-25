#include <stdio.h>

#include "../log_helper.h"

int main(){

	// Optional
	//set_absolute_path("/home/daniel/blabla/");
	start_log_file("my_benchmark", "size:x repetition:y");
	set_max_errors_iter(32);
	
	printf("log file is %s\n",get_log_file_name());
	
	int i;
	for(i = 0; i < 10; i++){

		start_iteration();
		// Execute the test (ONLY THE KERNEL), log functions will measure kernel time
		sleep(1);
		end_iteration();
		
		if(i%3==0){
		// Testing with error_count > 0 for some iterations
			// You can call as many log_error_detail(str) as you need
			log_error_detail("detail of error x");
			log_error_detail("detail of error y");

			// Tell log how many errors the iteration had
			log_error_count(i+1);
		} else{
			// If you have no errors you should tell. Really needed???
			log_error_count(0);
		}
	}

    end_log_file();
    
}
