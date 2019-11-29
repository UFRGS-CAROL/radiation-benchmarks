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
#define THRESHOLD_1 0x3A007358
#define ONE_BLOCK 1

//For NUMBER_PAR_PER_BOX iterations
//9e-05
#define THRESHOLD_FULL_BLOCK 5111808
#define FULL_BLOCK NUMBER_PAR_PER_BOX

//For AT_THE_END iterations
#define THRESHOLD_MAX 6141811
#define AT_THE_END_BLOCK NUMBER_PAR_PER_BOX + 2

void setup_double(Parameters& parameters, Log& log) {
	if (parameters.redundancy == NONE || parameters.generate) {
		UnhardenedKernelCaller<double> kc;
		setup_execution(parameters, log, kc);
	} else if (parameters.redundancy == DMR) {
		//CASE FOR 1 Iteration-------------------
		DMRKernelCaller<double> kc;
		setup_execution(parameters, log, kc);
	} else if (parameters.redundancy == DMRMIXED) {
		switch (parameters.block_check) {
		case ONE_BLOCK: {
			//CASE FOR 1 Iteration-------------------
			DMRMixedKernelCaller<ONE_BLOCK, float, double> kc(THRESHOLD_1);
			setup_execution(parameters, log, kc);

			break;
		}
			//---------------------------------------

		default:
			//CASE AT THE END Iterations----------------
			DMRMixedKernelCaller<AT_THE_END_BLOCK, float, double> kc(
			THRESHOLD_MAX);
			setup_execution(parameters, log, kc);

		}
	}
}
