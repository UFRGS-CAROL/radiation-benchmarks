/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "setup_template.h"
#include "setup_double.h"
#include "KernelCaller.h"

#include "Parameters.h"
#include "Log.h"

/**
 * Define the threshold to use on
 * the comparison method
 */
//For 1 iteration
#define THRESHOLD_1 5

//For 10 iterations
#define THRESHOLD_10 10

//For 100 iterations
#define THRESHOLD_100 98

//For MAX_LAVA iterations
#define THRESHOLD_MAX 500

void setup_double(Parameters& parameters, Log& log) {

	switch (parameters.redundancy) {
	case NONE: {
		UnhardenedKernelCaller<double> kc;
		setup_execution(parameters, log, kc);
		break;
	}
	case DMR: {
		switch (parameters.block_check) {
		case 1: {
			//CASE FOR 1 Iteration-------------------
			DMRKernelCaller<1, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------
		case 10: {
			//CASE FOR 10 Iterations-----------------
			DMRKernelCaller<10, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		case 100: {
			//CASE FOR 100 Iterations----------------
			DMRKernelCaller<100, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		case NUMBER_PAR_PER_BOX: {
			//CASE FOR 100 Iterations----------------
			DMRKernelCaller<NUMBER_PAR_PER_BOX, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		default:
			error(
					std::to_string(parameters.block_check)
							+ " operation check block not supported");
		}
		break;
	}
	case DMRMIXED:
		switch (parameters.block_check) {
		case 1: {
			//CASE FOR 1 Iteration-------------------
			DMRMixedKernelCaller<1, THRESHOLD_1, float, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------
		case 10: {
			//CASE FOR 10 Iterations-----------------
			DMRMixedKernelCaller<10, THRESHOLD_10, float, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		case 100: {
			//CASE FOR 100 Iterations----------------
			DMRMixedKernelCaller<100, THRESHOLD_100, float, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		case NUMBER_PAR_PER_BOX: {
			//CASE FOR 100 Iterations----------------
			DMRMixedKernelCaller<NUMBER_PAR_PER_BOX, THRESHOLD_MAX, float,
					double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		default:
			error(
					std::to_string(parameters.block_check)
							+ " operation check block not supported");
		}
		break;
	}
}
