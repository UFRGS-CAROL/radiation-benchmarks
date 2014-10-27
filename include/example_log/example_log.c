#include <stdio.h>

#include "../log_helper.h"

int main(){

	// Optional
	//set_absolute_path("/home/daniel/blabla/");
	start_log_file("my_benchmark");
	int i;
	for(i = 0; i < 10; i++){
		start_iteration();

		// Execute the test

		log_string("Some string that we may need to log");
		log_string_int("Some string that we may need to log with int: ",i);
		log_time(1.5);
		log_error(i);
	}

	printf("log file written in %s\n",get_log_file_name());
}
