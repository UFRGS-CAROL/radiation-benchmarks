// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    test_threaded.cpp:  Simple example using pre-trained model to test mojo
//    cnn with OpenMP multi-threading.
//
//    Instructions: 
//	  Add the "mojo" folder in your include path.
//    Download MNIST data and unzip locally on your machine:
//		(http://yann.lecun.com/exdb/mnist/index.html)
//    Download CIFAR-10 data and unzip locally on your machine:
//		(http://www.cs.toronto.edu/~kriz/cifar.html)
//    Set the data_path variable in the code to point to your data location.
// ==================================================================== mojo ==

#include <iostream> // cout
#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
//#include <tchar.h>

// define MOJO_OMP before loading the mojo cnn header or include the definition in the header
#define MOJO_OMP
#include <mojo.h>

#ifdef TIMING
long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
#endif

//*
#include "mnist_parser.h"
using namespace mnist;
//std::string data_path = "./data/";
//std::string model_file = "./models/mnist_deepcnet.mojo";
/*
#include "cifar_parser.h"
using namespace cifar;
std::string data_path="../data/cifar-10-batches-bin/";
std::string model_file="../models/cifar_deepcnet.mojo";
*/

int test(mojo::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels, int input_size)
{
	int out_size = cnn.out_size(); // we know this to be 10 for MNIST and CIFAR
	int correct_predictions = 0;

	int size = (int)test_images.size()/input_size;
	// use progress object for simple timing and status updating
	//mojo::progress progress(size, "  testing : ");

	const int record_cnt = size;
	// use standard omp parallel for loop, the number of threads determined by network.enable_omp() call
#pragma omp parallel
#pragma omp for reduction(+:correct_predictions) schedule(dynamic)  // dynamic schedule just helps the progress class to work correcly
	for (int k = 0; k < record_cnt; k++)
	{
		//if(k>1 && k<500){
		//    const int prediction = cnn.predict_class(test_images[k-1].data());
		//    if (prediction == test_labels[k]) correct_predictions++;
		//}else{
		// predict_class returnes the output index of the highest response
		const int prediction = cnn.predict_class(test_images[k].data());
		if (prediction == test_labels[k]) correct_predictions++;
		//}
		//if (k % 1000 == 0) progress.draw_progress(k);
	}
	//float dt = progress.elapsed_seconds();
	//std::cout << "  test time: " << dt << " seconds                                          " << std::endl;
	std::cout << "  records: " << record_cnt << std::endl;
	//std::cout << "  speed: " << (float)record_cnt / dt << " records/second" << std::endl;
	std::cout << "  accuracy: " << (float)correct_predictions / record_cnt*100.f << "%" << std::endl;
	return correct_predictions;
}

bool compareFiles(std::string& p1, std::string& p2) {
  std::ifstream f1(p1, std::ifstream::binary|std::ifstream::ate);
  std::ifstream f2(p2, std::ifstream::binary|std::ifstream::ate);

  if (f1.fail() || f2.fail()) {
    return false; //file problem
  }

  if (f1.tellg() != f2.tellg()) {
    return false; //size mismatch
  }

  //seek back to beginning and use std::equal to compare contents
  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);
  return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()), \
                    std::istreambuf_iterator<char>(), \
                    std::istreambuf_iterator<char>(f2.rdbuf()));
}


std::string data_path;
std::string model_file;
int main(int argc, char** argv)
{
#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int size, iterations, gold_correct_predictions;
    char *goldFile, *dataFolder, *modelFile;
    if (argc == 7) {
        size = atoi(argv[1]);
        goldFile = argv[2];
        gold_correct_predictions = atoi(argv[3]);
        dataFolder = argv[4];
        modelFile = argv[5];
        iterations = atoi(argv[6]);
	data_path = std::string(dataFolder);
	model_file = std::string(modelFile);
	if(size < 1 || size > 3){
        	fprintf(stderr, "Usage: %s <input size> <gold file> <gold # correct predictions> <input data folder> <model file> <#iterations>\n", argv[0]);
        	fprintf(stderr, "\t<input size> should be 1, 2 , or 3; 1 the biggest, 3 the smallest\n", argv[0]);
        	exit(1);
	}
    } else {
        fprintf(stderr, "Usage: %s <input size> <gold file> <gold # correct predictions> <input data folder> <model file> <#iterations>\n", argv[0]);
        fprintf(stderr, "\t<input size> should be 1, 2 , or 3; 1 the biggest, 3 the smallest\n", argv[0]);
        exit(1);
    }
	// == parse data
	// array to hold image data (note that mojo cnn does not require use of std::vector)
	std::vector<std::vector<float>> test_images;
	// array to hold image labels 
	std::vector<int> test_labels;
	// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
	if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}
	// == setup the network 
	mojo::network cnn; 
	// here we need to prepare mojo cnn to store data from multiple threads
	// !! enable_omp must be set prior to loading or creating a model !!
	cnn.enable_omp(); 
	// load model 
	if (!cnn.read(model_file)) { std::cerr << "error: could not read model.\n"; return 1; }
	std::cout << "Mojo Configuration:" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[200];
    snprintf(test_info, 200, "size:%d", size);
    start_log_file("openmpMojoCNN", test_info);
#endif

#ifdef TIMING
    setup_end = timing_get_time();
#endif
    int loop;
    for(loop=0; loop<iterations; loop++) {
#ifdef TIMING
        loop_start = timing_get_time();
#endif
#ifdef TIMING
        kernel_start = timing_get_time();
#endif
#ifdef LOGS
        start_iteration();
#endif
	// == run the test 
	std::cout << "Testing " << data_name() << ":" << std::endl;
	// this function will loop through all images, call predict, and print out stats
	int correct_predictions = test(cnn, test_images, test_labels, size);
#ifdef LOGS
        end_iteration();
#endif
#ifdef TIMING
        kernel_end = timing_get_time();
#endif
#ifdef TIMING
        check_start = timing_get_time();
#endif
	int errors = 0;
	if(correct_predictions != gold_correct_predictions){
                errors++;
                char error_detail[200];
                sprintf(error_detail," correct_predictions, r: %d, e: %d", correct_predictions, gold_correct_predictions);
#ifdef LOGS
                log_error_detail(error_detail);
#endif
	}
//	char * net_name = log_name();
//	std::ofstream temp(net_name, std::ios::binary);
//	cnn.write(temp, true, false);
//	if(!compareFiles(std::string(net_name),std::string(goldFile))){
//                errors++;
//                char error_detail[200];
//                sprintf(error_detail," network different, corrupted network file: %s", net_name);
//#ifdef LOGS
//                log_error_detail(error_detail);
//#endif
//	}
//#ifdef LOGS
//        log_error_count(errors);
//#endif
	errors =+ cnn.check_gold(goldFile);
#ifdef TIMING
        check_end = timing_get_time();
#endif
	if(errors > 0){
		printf("Errors: %d\n",errors);
		cnn.clear();
		// calls MNIST::parse_test_data  or  CIFAR10::parse_test_data depending on 'using'
		if(!parse_test_data(data_path, test_images, test_labels)) {std::cerr << "error: could not parse data.\n"; return 1;}
		cnn.enable_omp(); 
		// load model 
		if (!cnn.read(model_file)) { std::cerr << "error: could not read model.\n"; return 1; }
	} else {
		printf(".");
	}
#ifdef TIMING
        loop_end = timing_get_time();
        double setup_timing = (double) (setup_end - setup_start) / 1000000;
        double loop_timing = (double) (loop_end - loop_start) / 1000000;
        double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
        double check_timing = (double) (check_end - check_start) / 1000000;
        printf("\n\tTIMING:\n");
        printf("setup: %f\n",setup_timing);
        printf("loop: %f\n",loop_timing);
        printf("kernel: %f\n",kernel_timing);
        printf("check: %f\n",check_timing);
#endif

    }
#ifdef LOGS
    end_log_file();
#endif

	cnn.clear();
	std::cout << std::endl;
	return 0;
}
