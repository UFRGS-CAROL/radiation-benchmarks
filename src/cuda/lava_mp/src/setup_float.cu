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
#include "setup.h"
#include "KernelCaller.h"

#include "Parameters.h"
#include "Log.h"

/**
 * Define the threshold to use on
 * the comparison method
 */
//For 1 iteration
//00000000 01001100 00000000 00000000
#define THRESHOLD_1 5111808
#define ONE_BLOCK 1

//For 10 iterations
//3e-05
#define THRESHOLD_12 67108864
#define TWELVE_BLOCK 12

//For 100 iterations
//9e-05
#define THRESHOLD_96 5111808
#define NINETY_BLOCK 96

//For MAX_LAVA iterations
//4e-03
#define THRESHOLD_MAX 4194304
#define MAX_BLOCK NUMBER_PAR_PER_BOX

void setup_float(Parameters& parameters, Log& log) {

	switch (parameters.redundancy) {
//	case NONE: {
//		UnhardenedKernelCaller<float> kc;
//		setup_execution(parameters, log, kc);
//		break;
//	}
	/*
	case DMR: {
		switch (parameters.block_check) {
		case ONE_BLOCK: {
			//CASE FOR 1 Iteration-------------------
			DMRKernelCaller<ONE_BLOCK, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------
		case TWELVE_BLOCK: {
			//CASE FOR 10 Iterations-----------------
			DMRKernelCaller<TWELVE_BLOCK, double> kc;
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

//		case NINETY_BLOCK: {
//			//CASE FOR 100 Iterations----------------
//			DMRKernelCaller<NINETY_BLOCK, double> kc;
//			setup_execution(parameters, log, kc);
//
//			break;
//		}
			//---------------------------------------

		case MAX_BLOCK: {
			//CASE FOR 100 Iterations----------------
			DMRKernelCaller<MAX_BLOCK, double> kc;
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
	}*/
	case DMRMIXED:
		switch (parameters.block_check) {
//		case ONE_BLOCK: {
//			//CASE FOR 1 Iteration-------------------
//			DMRMixedKernelCaller<ONE_BLOCK, double, float> kc(THRESHOLD_1);
//			setup_execution(parameters, log, kc);
//
//			break;
//		}
			//---------------------------------------
//		case TWELVE_BLOCK: {
//			//CASE FOR 10 Iterations-----------------
//			DMRMixedKernelCaller<TWELVE_BLOCK, THRESHOLD_12, float, double> kc;
//			setup_execution(parameters, log, kc);
//
//			break;
//		}
			//---------------------------------------
//
//		case NINETY_BLOCK: {
//			//CASE FOR 100 Iterations----------------
//			DMRMixedKernelCaller<NINETY_BLOCK, THRESHOLD_96, float, double> kc;
//			setup_execution(parameters, log, kc);
//
//			break;
//		}
			//---------------------------------------
//
//		case MAX_BLOCK: {
//			//CASE FOR 100 Iterations----------------
//			DMRMixedKernelCaller<MAX_BLOCK, THRESHOLD_MAX, float, double> kc;
//			setup_execution(parameters, log, kc);
//
//			break;
//		}
			//---------------------------------------

		default:
			error(
					std::to_string(parameters.block_check)
							+ " operation check block not supported");
		}
		break;
	}
}
