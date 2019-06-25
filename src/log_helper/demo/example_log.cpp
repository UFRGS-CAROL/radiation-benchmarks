#include <iostream> //For cout

#include <unistd.h> //For sleep

#include "log_helper.hpp"

int main(){

    // Start the log with filename including "my_benchmark" to its name, and
    // header inside log file will print the detail "size:x repetition:y"
    LogHelper log((char*)"my_benchmark", (char*)"size:x repetition:y");

    // set the maximum number of errors allowed for each iteration,
    // default is 500
    log.set_max_errors_iter(32);

    // set the interval of iteration to print details of current test, 
    // default is 1
    log.set_iter_interval_print(5);
    
    std::cout << "log file is " << log.get_log_file_name() << "\n";
    
    int i;
    for(i = 0; i < 40; i++){

        log.start_iteration();
        // Execute the test (ONLY THE KERNEL), log functions will measure kernel time
        sleep(1);
        log.end_iteration();


        int error_count = 0;
        int info_count = 0;
        
        if(i%8==0){
            // Testing with error_count > 0 for some iterations
            // You can call as many log_error_detail(str) as you need
            // However, it will log only the 500 errors or the
            // max_errors_iter set with set_max_errors_iter()
            log.log_error_detail("detail of error x");

            // Tell log how many errors the iteration had
            error_count = i+1;
        }
        if (i%16==0){
            log.log_info_detail("info of event during iteration");
            info_count = info_count + 520;
        }

        // log how many errors the iteration had
        // if error_count is greater than 500, or the
        // max_errors_iter set with set_max_errors_iter()
        // it will terminate the execution
        log.log_error_count(error_count);
        log.log_info_count(info_count);
    }

    // Finish the log file
    // not necessary for CPP :)
}
