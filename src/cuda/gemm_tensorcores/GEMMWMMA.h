/*
 * GEMMWMMA.h
 *
 *  Created on: 12/08/2018
 *      Author: fernando
 */

#ifndef GEMMWMMA_H_
#define GEMMWMMA_H_

#include <type_traits>


// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

namespace radiation {


//ERROR functions definitions
#define check_framework_errors(error) __check_framework_errors(error, __LINE__, __FILE__)
#define error(error) __error(error, __LINE__, __FILE__)

void __check_framework_errors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

void __error(const char* error, int line, const char* file) {
	printf("%s - Line: %d at %s\n", error, line, file);
	exit (EXIT_FAILURE);
}

template<class host_half_t, class half_t, class real_t>
class GEMMWMMA {

public:

	// Memory pointers to device and host data
	half_t* device_ptr_a = nullptr;
	half_t* device_ptr_b = nullptr;
	real_t* device_ptr_c = nullptr;
	real_t* device_ptr_d = nullptr;

	// Size of the matrix
	size_t cols_a, rows_a;
	size_t cols_b, rows_b;
	size_t cols_c, rows_c;

	size_t byte_size_c;

	GEMMWMMA(const host_half_t* host_ptr_a, const host_half_t* host_ptr_b,
			const host_half_t* host_ptr_c, size_t rows_a, size_t cols_a,
			size_t cols_b);

	virtual ~GEMMWMMA();
	/**
	 * Template multiplication
	 */
	void mul();

	void push_arrays(const host_half_t* host_ptr_a,
			const host_half_t* host_ptr_b, const host_half_t* host_ptr_c);


	void pull_array(host_half_t* host_ptr_d);

};

} /* namespace radiation */

#endif /* GEMMWMMA_H_ */
