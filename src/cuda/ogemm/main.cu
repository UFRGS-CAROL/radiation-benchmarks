#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>      // std::ifstream
#include <sstream>      // std::stringstream

#include <iomanip>
#include <limits>


#ifdef OMP
#include <omp.h>
#endif

#include "half.hpp"
#include "Log.h"
#include "kernels.h"

#ifndef DEFAULT_INPUT_SIZE
#define DEFAULT_INPUT_SIZE 8192
#endif

#define M 8192
#define N 8192
#define K 8192

#define GENERATOR_MAXABSVALUE 2.0
#define GENERATOR_MINABSVALUE 0


#define ZERO_HALF 6.5469




typedef half_float::half host_half;
typedef std::vector<host_half> half_vector;



template<class real_t>
void write_gold_to_file(std::string gold_path, std::vector<real_t>& gold) {
	std::ofstream f_gold(gold_path, std::ofstream::out | std::ofstream::binary);
	if (f_gold.is_open()) {
		f_gold.write(reinterpret_cast<char*>(gold.data()),
				sizeof(real_t) * gold.size());
		f_gold.close();
	} else {
		throw std::runtime_error("Could not write gold file\n");
	}
}

template<class real_t> int is_output_ok(std::vector<real_t>& d0,
		std::vector<real_t>& d1, std::vector<real_t>& d2,
		std::vector<real_t>& correct_vector) {

	int memory_errors = 0;
	for (size_t i = 0; i < d0.size(); i++) {
		real_t val_output0 = d0[i];
		real_t val_output1 = d1[i];
		real_t val_output2 = d2[i];
		real_t val_output = val_output0;

		if ((val_output0 != val_output1) || (val_output0 != val_output2)) {
			memory_errors++;

			if ((val_output0 != val_output1) && (val_output1 != val_output2)
					&& (val_output0 != val_output2)) {
				// All 3 values diverge
				memory_errors++;
			} else if (val_output1 == val_output2) {
				// Only value 0 diverge
				val_output = val_output1;
			} else if (val_output0 == val_output2) {
				// Only value 1 diverge
				val_output = val_output0;
			} else if (val_output0 == val_output1) {
				// Only value 2 diverge
				val_output = val_output0;
			}
		}
		correct_vector[i] = val_output;
	}
	return memory_errors;
}

template<class real_t> void retrieve_matrices(half_vector& a_host_vector,
		half_vector& b_host_vector, std::vector<real_t>& c_host_vector,
		std::vector<real_t>& gold_host_vector, Log& log) {

	double start = log.mysecond();
	std::ifstream f_a(log.a_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_b(log.b_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_c(log.c_input_path, std::ios::in | std::ios::binary);
	std::ifstream f_gold(log.gold_inout_path,
			std::ifstream::in | std::ifstream::binary);

	if (f_a.is_open() && f_b.is_open() && f_c.is_open() && f_gold) {

		f_a.seekg(0, std::ios::beg);
		f_a.read(reinterpret_cast<char*>(a_host_vector.data()),
				sizeof(host_half) * a_host_vector.size());

		f_b.seekg(0, std::ios::beg);
		f_b.read(reinterpret_cast<char*>(b_host_vector.data()),
				sizeof(host_half) * b_host_vector.size());

		f_c.seekg(0, std::ios::beg);
		f_c.read(reinterpret_cast<char*>(c_host_vector.data()),
				sizeof(real_t) * c_host_vector.size());

		f_gold.seekg(0, std::ios::beg);
		f_gold.read(reinterpret_cast<char*>(gold_host_vector.data()),
				sizeof(real_t) * gold_host_vector.size());

		f_a.close();
		f_b.close();
		f_c.close();
		f_gold.close();
	} else {
		log.log_error("Could not retrieve the matrices");
		throw std::runtime_error("Could not retrieve the matrices\n");
	}

	std::cout << "Done with reading matrices " << log.mysecond() - start
			<< "s\n";
}

template<class host_real_t>
bool cmp(const host_real_t lhs, const host_real_t rhs) {
	const host_real_t diff = abs(lhs - rhs);

	std::cout << "d0= " << lhs << "d1 = " << rhs << std::endl;	
	const host_real_t zero = host_real_t(ZERO_HALF);
	if (diff > zero) {
		return false;
	}
	return true;
}


template<class real_t>
std::pair<int, int> check_output_errors(std::vector<real_t>& gold,  std::vector<real_t>& d0, std::vector<real_t>& d1, Log& log) {
	int host_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		real_t valGold = gold[i];
		real_t valOutput0 = d0[i];
		real_t valOutput1 = d1[i];

		// if (valGold != valOutput1 || !cmp(valOutput0, valOutput1)) {
		if (!cmp(valOutput0, valOutput1)){
					std::stringstream error_detail("");
					error_detail << "p: [" << int(floor(i / log.size_matrices))
							<< ", " << i % log.size_matrices << "], r: "
							<< valOutput1 << ", e: " << valGold << " smaller_precision: " << valOutput0;

					if (log.verbose && (host_errors < 10))
						std::cout << error_detail.str() << std::endl;

					log.log_error(error_detail.str());
					host_errors++;
		}
	}
	log.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	std::pair<int, int> res(0, host_errors);
	return res;
}

template<class host_real_t, class real_t, class half_t>
void call_mxm(half_vector& host_matrix_a, half_vector& host_matrix_b, half_vector& host_matrix_c,
	 half_vector& host_matrix_c_inc, Log& log_obj) {
	cudaEvent_t start, stop;
	float elapsedTime;

	//TODO
	generate_matrices_files<host_real_t>(host_matrix_a, host_matrix_b,log_obj);
	



	rad::DeviceVector<real_t> device_c(host_matrix_c), device_a(host_matrix_a), device_b(
			host_matrix_c);
	rad::DeviceVector<half_real_t> device_c_inc(host_matrix_c_inc);

	int tries = 0;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	cudaStream_t st;
	cudaStreamCreate(&st);	
	assert(M > 512 && N > 512 && M % 64 == 0 && N % 16 == 0 && K % 16 == 0);
	for (int it = 0; it < log_obj.iterations; it++) {
		double start_computation = log_obj.mysecond();
		log_obj.start_iteration_app();
				

		//TODO
		gemm_dmr();
			
	
		log_obj.end_iteration_app();
		double end_computation = log_obj.mysecond();

		host_c = device_c.to_vector();
		host_c_inc = device_c_inc.to_vector();

		cudaEventCreate(&stop);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsedTime, start,stop);
		printf("Elapsed time : %f ms\n" ,elapsedTime);

	    if (!log_obj.generate) {
			//fault test 
//			if(it == 2){
//			host_matrix_d0[2]= (real_t) 5.00;
//			}
			//
			std::pair<int, int> errors;
			double start, end;
		
			start = log_obj.mysecond();
		    //errors = compare_output_matrices(host_gold, host_matrix_d0, log_obj);
			//printf("%f\n", host_matrix_d0[0]);
			
			//TODO
			errors = check_output_errors(host_gold, host_matrix_d0, host_matrix_d1,log_obj);
			end = log_obj.mysecond();
			
			std::cout << "Iteration: " << it << " memory errors "
					<< errors.first << " radiation errors " << errors.second
					<< ". Time spent on computation " << end_computation - start_computation
					<< "s. Time spent on comparing " << end - start << "s."
					<< std::endl;

			//If errors != 0 reload matrices to gpu
			if (errors.first != 0 || errors.second != 0) {
				rad::DeviceVector<real_t> device_c(host_matrix_c), device_a(host_matrix_a), device_b(
				host_matrix_b);
			}

		}
		
	}
	cudaStreamDestroy(st);
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start,stop);
	printf("time : %f s\n" ,(elapsedTime/1000));

	if (log_obj.generate) {			

		write_gold_to_file<host_real_t>(log_obj.gold_inout_path, host_matrix_d1);	
	}

   // host_real_t largest = host_matrix_d1[0];
   // for(int z = 1;z <(log_obj.size_matrices * log_obj.size_matrices) ; z++) {

   //    if(largest < host_matrix_d1[z])
   //       largest = host_matrix_d1[z];
       
   // } 
  

   // host_real_t lowest = host_matrix_d0[0];
   // for(int z = 1;z <(log_obj.size_matrices * log_obj.size_matrices) ; z++) {

   //    if(lowest < host_matrix_d0[z])
   //       lowest = host_matrix_d0[z];
     
   // } 
   // std::cout << "Largest element in array is: " << largest <<std::endl;
   // std::cout << "lowest element in array is: " << lowest<<std::endl;

   // std::cout << "treshold is: " <<(largest-lowest)<<std::endl;


}   


void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --gold <gold file, DEFAULT=./gold.matrix > --size <matrix size, DEFAULT=8192> "
					"--iterations <how many iterations, optional> --input_a <input A, DEFAUL=./input_a.matrix> "
					"--input_b <input B, DEFAUL=./input_b.matrix> --input_c <input C, DEFAUL=./input_c.matrix>  --precision <float/double, DEFAULT=float>"
			<< std::endl;
}

int main(int argc, char** argv) {
	Log log_obj(argc, argv, DEFAULT_INPUT_SIZE);

	std::cout << "Generate: " << log_obj.generate << std::endl;
	std::cout << "A input path: " << log_obj.a_input_path << std::endl;
	std::cout << "B input path: " << log_obj.b_input_path << std::endl;
	std::cout << "C input path: " << log_obj.c_input_path << std::endl;
	std::cout << "Gold in/out path: " << log_obj.gold_inout_path << std::endl;
	std::cout << "Iterations: " << log_obj.iterations << std::endl;
	std::cout << "Matrix size: " << log_obj.size_matrices << std::endl;
	std::cout << "Precision: " << log_obj.precision << std::endl;
	std::cout << "Verbose: " << log_obj.verbose << std::endl;

	
	int m;
	int n;
	int k;	
	m = n = k = log_obj.size_matrices;
	int lda = m;
	int ldb = n;
	int ldc = k;
	real_t alpha = real_t(1.1f);
	real_t beta = real_t(1.2f);
	const std::vector<real_t> zero_vector(m * k, 0.0);
	const std::vector<half_real_t> zero_vector_inc(m * k, 0.0);
	std::vector<real_t> host_a(m * n, 2.0);
	std::vector<real_t> host_b(n * k, 2.0);
	std::vector<real_t> host_c(m * k, 0.0);
	std::vector<half_real_t> host_c_inc(m * k, 0.0);



	// TODO 
	if (log_obj.precision == "float") {
		call_mxm<float, float, half>(host_a, host_b, host_c, host_c_inc, log_obj);
	}
	if (log_obj.precision == "double") {
		call_mxm<double, double, half>(host_a, host_b, host_c, host_c_inc, log_obj);
	}	
	if (log_obj.precision == "mixed") {
		
		call_mxm<double, float, half>(host_a, host_b, host_c, host_c_inc, log_obj);
	}

	std::cout << "Finished computation\n";
	return 0;
}
