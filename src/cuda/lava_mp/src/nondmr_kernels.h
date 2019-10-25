/*
 * nondmr_kernels.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef NONDMR_KERNELS_H_
#define NONDMR_KERNELS_H_

#include "device_funtions.h"
#include "types.h"

//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------
// #if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
template<typename real_t>
__global__ void kernel_gpu_cuda_nondmr(par_str<real_t> d_par_gpu, dim_str d_dim_gpu,
		box_str* d_box_gpu, FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
		FOUR_VECTOR<real_t>* d_fv_gpu) {

	//---------------------------------------------------------------------
	//	THREAD PARAMETERS
	//---------------------------------------------------------------------

	int bx = blockIdx.x;		 // get current horizontal block index (0-n)
	int tx = threadIdx.x;		 // get current horizontal thread index (0-n)
	int wtx = tx;

	//---------------------------------------------------------------------
	//	DO FOR THE NUMBER OF BOXES
	//---------------------------------------------------------------------

	if (bx < d_dim_gpu.number_boxes) {

		//-------------------------------------------------------------
		//	Extract input parameters
		//-------------------------------------------------------------

		// parameters
		real_t a2 = real_t(2.0) * d_par_gpu.alpha * d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR<real_t> *rA;
		FOUR_VECTOR<real_t> *fA;
		__shared__ FOUR_VECTOR<real_t> rA_shared[200];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR<real_t> *rB;
		real_t *qB;
		int j = 0;
		__shared__ FOUR_VECTOR<real_t> rB_shared[200];
		__shared__ real_t qB_shared[200];

		// common
		real_t r2;
		real_t u2;
		real_t vij;
		real_t fs;
		real_t fxij;
		real_t fyij;
		real_t fzij;
		THREE_VECTOR<real_t> d;

		//-------------------------------------------------------------
		//	Home box
		//-------------------------------------------------------------

		//-------------------------------------------------------------
		//	Setup parameters
		//-------------------------------------------------------------

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//-------------------------------------------------------------
		//	Copy to shared memory
		//-------------------------------------------------------------

		// home box - shared memory
		while (wtx < NUMBER_PAR_PER_BOX) {
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//-------------------------------------------------------------
		//	nei box loop
		//-------------------------------------------------------------

		// loop over neiing boxes of home box
		for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

			//---------------------------------------------
			//	nei box - get pointer to the right box
			//---------------------------------------------

			if (k == 0) {
				pointer = bx;	 // set first box to be processed to home box
			} else {
				// remaining boxes are nei boxes
				pointer = d_box_gpu[bx].nei[k - 1].number;
			}

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - shared memory
			while (wtx < NUMBER_PAR_PER_BOX) {
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//-----------------------------------------------------
			//	Calculation
			//-----------------------------------------------------
			while (wtx < NUMBER_PAR_PER_BOX) {

				for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

					r2 = rA_shared[wtx].v + rB_shared[j].v -
					DOT(
							rA_shared[wtx],
							rB_shared[j]
					);

					u2 = a2 * r2;
					vij= exp__(-u2);

					fs = real_t(2.0) * vij;

					d.x = rA_shared[wtx].x - rB_shared[j].x;

					fxij = fs * d.x;

					d.y = rA_shared[wtx].y - rB_shared[j].y;

					fyij = fs * d.y;

					d.z = rA_shared[wtx].z - rB_shared[j].z;

					fzij = fs * d.z;

					fA[wtx].v += (real_t)(qB_shared[j] * vij);
					fA[wtx].x += (real_t)(qB_shared[j] * fxij);
					fA[wtx].y += (real_t)(qB_shared[j] * fyij);
					fA[wtx].z += (real_t)(qB_shared[j] * fzij);
				}
				// increment work thread index
				wtx = wtx + NUMBER_THREADS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140
		}
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
	}
}



#endif /* NONDMR_KERNELS_H_ */
