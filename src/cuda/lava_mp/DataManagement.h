/*
 * DataManagement.h
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#ifndef DATAMANAGEMENT_H_
#define DATAMANAGEMENT_H_

#ifdef USE_OMP
#include <omp.h>
#endif

#include <vector>
#include <cuda_runtime.h>

#include "cuda_utils.h"

template<typename full>
class DataManagement {
	Parameters& setup_parameters;
	std::vector<cudaStream_t> streams;

public:
	DataManagement(Parameters& setup_parameters) :
			setup_parameters(setup_parameters) {
//		for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//			//=====================================================================
//			//	GPU SETUP MEMORY
//			//=====================================================================
//
//			//==================================================
//			//	boxes
//			//==================================================
//			checkFrameworkErrors(
//					cudaMalloc((void ** )&(d_box_gpu[streamIdx]), dim_cpu.box_mem));
//			//==================================================
//			//	rv
//			//==================================================
//			checkFrameworkErrors(
//					cudaMalloc((void ** )&(d_rv_gpu[streamIdx]),
//							dim_cpu.space_mem));
//			//==================================================
//			//	qv
//			//==================================================
//			checkFrameworkErrors(
//					cudaMalloc((void ** )&(d_qv_gpu[streamIdx]),
//							dim_cpu.space_mem2));
//
//			//==================================================
//			//	fv
//			//==================================================
//			checkFrameworkErrors(
//					cudaMalloc((void ** )&(d_fv_gpu[streamIdx]),
//							dim_cpu.space_mem));
//
//			//=====================================================================
//			//	GPU MEMORY			COPY
//			//=====================================================================
//
//			//==================================================
//			//	boxes
//			//==================================================
//
//			checkFrameworkErrors(
//					cudaMemcpy(d_box_gpu[streamIdx], box_cpu, dim_cpu.box_mem,
//							cudaMemcpyHostToDevice));
//			//==================================================
//			//	rv
//			//==================================================
//
//			checkFrameworkErrors(
//					cudaMemcpy(d_rv_gpu[streamIdx], rv_cpu, dim_cpu.space_mem,
//							cudaMemcpyHostToDevice));
//			//==================================================
//			//	qv
//			//==================================================
//
//			checkFrameworkErrors(
//					cudaMemcpy(d_qv_gpu[streamIdx], qv_cpu, dim_cpu.space_mem2,
//							cudaMemcpyHostToDevice));
//			//==================================================
//			//	fv
//			//==================================================
//
//			// This will be done with memset at the start of each iteration.
//			// checkFrameworkErrors( cudaMemcpy( d_fv_gpu[streamIdx], fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
//		}
//
//		//==================================================
//		//	fv_gold for GoldChkKernel
//		//==================================================
//		if (gpu_check) {
//			checkFrameworkErrors(
//					cudaMalloc((void** )&d_fv_gold_gpu, dim_cpu.space_mem));
//			checkFrameworkErrors(
//					cudaMemcpy(d_fv_gold_gpu, fv_cpu_GOLD, dim_cpu.space_mem2,
//							cudaMemcpyHostToDevice));
//		}

//=====================================================================
//	STREAMS
//=====================================================================
		for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams;
				streamIdx++) {
			checkFrameworkErrors(
					cudaStreamCreateWithFlags(&(this->streams[streamIdx]),
							cudaStreamNonBlocking));
		}

	}
	virtual ~DataManagement() {
//
//		//=====================================================================
//		//	GPU MEMORY DEALLOCATION
//		//=====================================================================
//		for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//			cudaFree(d_rv_gpu[streamIdx]);
//			cudaFree(d_qv_gpu[streamIdx]);
//			cudaFree(d_fv_gpu[streamIdx]);
//			cudaFree(d_box_gpu[streamIdx]);
//		}
//		if (gpu_check) {
//			cudaFree(d_fv_gold_gpu);
//		}
		//Only destroy stream, the others will be automatically destroyed
		for (auto stream : this->streams) {
			checkFrameworkErrors(cudaStreamDestroy(stream));
		}
	}

	void generateInput() {
		// random generator seed set to random value - time in this case
//		FILE *fp;
//		int i;
//
//		printf("Generating input...\n");
//
//		srand(time(NULL));
//
//		// input (distances)
//		if ((fp = fopen(input_distances, "wb")) == 0) {
//			printf("The file 'input_distances' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//		*rv_cpu = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			// get a number in the range 0.1 - 1.0
//			(*rv_cpu)[i].v = (tested_type_host)(rand() % 10 + 1) / 10.0;
//			fwrite(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
//			// get a number in the range 0.1 - 1.0
//			(*rv_cpu)[i].x = (tested_type_host)(rand() % 10 + 1) / 10.0;
//			fwrite(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
//			// get a number in the range 0.1 - 1.0
//			(*rv_cpu)[i].y = (tested_type_host)(rand() % 10 + 1) / 10.0;
//			fwrite(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
//			// get a number in the range 0.1 - 1.0
//			(*rv_cpu)[i].z = (tested_type_host)(rand() % 10 + 1) / 10.0;
//			fwrite(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
//		}
//		fclose(fp);
//
//		// input (charge)
//		if ((fp = fopen(input_charges, "wb")) == 0) {
//			printf("The file 'input_charges' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//
//		*qv_cpu = (tested_type_host*) malloc(dim_cpu.space_mem2);
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			// get a number in the range 0.1 - 1.0
//			(*qv_cpu)[i] = (tested_type_host)(rand() % 10 + 1) / 10.0;
//			fwrite(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
//		}
//		fclose(fp);
	}

	void readInput() {
//		FILE *fp;
//		int i;
//		size_t return_value[4];
//		// size_t return_value;
//
//		// input (distances)
//		if ((fp = fopen(input_distances, "rb")) == 0) {
//			printf("The file 'input_distances' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//
//		*rv_cpu = (FOUR_VECTOR_HOST*) malloc(dim_cpu.space_mem);
//		if (*rv_cpu == NULL) {
//			printf("error rv_cpu malloc\n");
//	#ifdef LOGS
//			log_error_detail((char *)"error rv_cpu malloc"); end_log_file();
//	#endif
//			exit(1);
//		}
//
//		// return_value = fread(*rv_cpu, sizeof(FOUR_VECTOR_HOST), dim_cpu.space_elem, fp);
//		// if (return_value != dim_cpu.space_elem) {
//		// 	printf("error reading rv_cpu from file\n");
//		// 	#ifdef LOGS
//		// 		log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
//		// 	#endif
//		// 	exit(1);
//		// }
//
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			return_value[0] = fread(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
//			return_value[1] = fread(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
//			return_value[2] = fread(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
//			return_value[3] = fread(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
//			if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
//					|| return_value[3] == 0) {
//				printf("error reading rv_cpu from file\n");
//	#ifdef LOGS
//				log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
//	#endif
//				exit(1);
//			}
//		}
//		fclose(fp);
//
//		// input (charge)
//		if ((fp = fopen(input_charges, "rb")) == 0) {
//			printf("The file 'input_charges' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//
//		*qv_cpu = (tested_type_host*) malloc(dim_cpu.space_mem2);
//		if (*qv_cpu == NULL) {
//			printf("error qv_cpu malloc\n");
//	#ifdef LOGS
//			log_error_detail((char *)"error qv_cpu malloc"); end_log_file();
//	#endif
//			exit(1);
//		}
//
//		// return_value = fread(*qv_cpu, sizeof(tested_type_host), dim_cpu.space_elem, fp);
//		// if (return_value != dim_cpu.space_elem) {
//		// 	printf("error reading qv_cpu from file\n");
//		// 	#ifdef LOGS
//		// 		log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
//		// 	#endif
//		// 	exit(1);
//		// }
//
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			return_value[0] = fread(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
//			if (return_value[0] == 0) {
//				printf("error reading qv_cpu from file\n");
//	#ifdef LOGS
//				log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
//	#endif
//				exit(1);
//			}
//		}
//		fclose(fp);
//
//		// =============== Fault injection
//		if (fault_injection) {
//			(*qv_cpu)[2] = 0.732637263; // must be in range 0.1 - 1.0
//			printf("!!> Fault injection: qv_cpu[2]=%f\n", (double) (*qv_cpu)[2]);
//		}
		// ========================
	}

	void readGold() {
//		FILE *fp;
//		size_t return_value[4];
//		// size_t return_value;
//		int i;
//
//		if ((fp = fopen(output_gold, "rb")) == 0) {
//			printf("The file 'output_forces' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//
//		// return_value = fread(fv_cpu_GOLD, sizeof(FOUR_VECTOR_HOST), dim_cpu.space_elem, fp);
//		// if (return_value != dim_cpu.space_elem) {
//		// 	printf("error reading rv_cpu from file\n");
//		// 	#ifdef LOGS
//		// 		log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
//		// 	#endif
//		// 	exit(1);
//		// }
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			return_value[0] = fread(&(fv_cpu_GOLD[i].v), 1, sizeof(tested_type),
//					fp);
//			return_value[1] = fread(&(fv_cpu_GOLD[i].x), 1, sizeof(tested_type),
//					fp);
//			return_value[2] = fread(&(fv_cpu_GOLD[i].y), 1, sizeof(tested_type),
//					fp);
//			return_value[3] = fread(&(fv_cpu_GOLD[i].z), 1, sizeof(tested_type),
//					fp);
//			if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
//					|| return_value[3] == 0) {
//				printf("error reading rv_cpu from file\n");
//	#ifdef LOGS
//				log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
//	#endif
//				exit(1);
//			}
//		}
//		fclose(fp);
	}

	void writeGold() {
//		FILE *fp;
//		int i;
//
//		if ((fp = fopen(output_gold, "wb")) == 0) {
//			printf("The file 'output_forces' was not opened\n");
//			exit(EXIT_FAILURE);
//		}
//		int number_zeros = 0;
//		for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			if ((*fv_cpu)[i].v == tested_type_host(0.0))
//				number_zeros++;
//			if ((*fv_cpu)[i].x == tested_type_host(0.0))
//				number_zeros++;
//			if ((*fv_cpu)[i].y == tested_type_host(0.0))
//				number_zeros++;
//			if ((*fv_cpu)[i].z == tested_type_host(0.0))
//				number_zeros++;
//
//			fwrite(&((*fv_cpu)[i].v), 1, sizeof(tested_type), fp);
//			fwrite(&((*fv_cpu)[i].x), 1, sizeof(tested_type), fp);
//			fwrite(&((*fv_cpu)[i].y), 1, sizeof(tested_type), fp);
//			fwrite(&((*fv_cpu)[i].z), 1, sizeof(tested_type), fp);
//		}
//		fclose(fp);
	}

	// Returns true if no errors are found. False if otherwise.
	// Set votedOutput pointer to retrieve the voted matrix
	void checkOutputErrors() {
//		int host_errors = 0;
//
//	// #pragma omp parallel for shared(host_errors)
//		for (int i = 0; i < dim_cpu.space_elem; i = i + 1) {
//			FOUR_VECTOR_HOST valGold = fv_cpu_GOLD[i];
//			FOUR_VECTOR_HOST valOutput = fv_cpu[i];
//			if (valGold != valOutput) {
//	#pragma omp critical
//				{
//					char error_detail[500];
//					host_errors++;
//
//					snprintf(error_detail, 500,
//							"stream: %d, p: [%d], v_r: %1.20e, v_e: %1.20e, x_r: %1.20e, x_e: %1.20e, y_r: %1.20e, y_e: %1.20e, z_r: %1.20e, z_e: %1.20e\n",
//							streamIdx, i, (double) valOutput.v, (double) valGold.v,
//							(double) valOutput.x, (double) valGold.x,
//							(double) valOutput.y, (double) valGold.y,
//							(double) valOutput.z, (double) valGold.z);
//					if (verbose && (host_errors < 10))
//						printf("%s\n", error_detail);
//	#ifdef LOGS
//					if ((host_errors<MAX_LOGGED_ERRORS_PER_STREAM))
//					log_error_detail(error_detail);
//	#endif
//				}
//			}
//		}
//
//		// printf("numErrors:%d", host_errors);
//
//	#ifdef LOGS
//		log_error_count(host_errors);
//	#endif
//		if (host_errors != 0)
//			printf("#");
//
//		return (host_errors == 0);
	}

};

#endif /* DATAMANAGEMENT_H_ */
