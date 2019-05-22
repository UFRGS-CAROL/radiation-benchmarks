/*
 * lava_kernels.h
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#ifndef LAVA_KERNELS_H_
#define LAVA_KERNELS_H_

#include <cuda_fp16.h>


//=============================================================================
//	DEFINE / INCLUDE
//=============================================================================
#define NUMBER_PAR_PER_BOX 192	 // keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 192		 // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

// STABLE
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))
#define MAX_LOGGED_ERRORS_PER_STREAM 100

#define H2_DOT(A,B) (__hfma2((A.x), (B.x), __hfma2((A.y), (B.y), __hmul2((A.z), (B.z)))))


//__host__ __device__ inline bool operator==(const box_str& lhs,
//		const box_str& rhs) {
//	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z)
//			&& (lhs.number == rhs.number) && (lhs.offset == rhs.offset);
//}
//__host__ __device__ inline bool operator!=(const box_str& lhs,
//		const box_str& rhs) {
//	return !operator==(lhs, rhs);
//}



//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------
// #if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
template<typename full>
__global__ void kernel_gpu_cuda(par_str<full> d_par_gpu, dim_str d_dim_gpu,
		box_str* d_box_gpu, FOUR_VECTOR<full>* d_rv_gpu, full* d_qv_gpu,
		FOUR_VECTOR<full>* d_fv_gpu) {
/*
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
		tested_type a2 = tested_type(2.0) * d_par_gpu.alpha * d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR *rA;
		FOUR_VECTOR *fA;
		__shared__ FOUR_VECTOR rA_shared[200];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR *rB;
		tested_type *qB;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared[200];
		__shared__ tested_type qB_shared[200];

		// common
		tested_type r2;
		tested_type u2;
		tested_type vij;
		tested_type fs;
		tested_type fxij;
		tested_type fyij;
		tested_type fzij;
		THREE_VECTOR d;

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
#if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
					vij= exp(-u2);
#elif defined(PRECISION_HALF)
					vij= hexp(-u2);
#endif
					fs = tested_type(2.0) * vij;

					d.x = rA_shared[wtx].x - rB_shared[j].x;

					fxij = fs * d.x;

					d.y = rA_shared[wtx].y - rB_shared[j].y;

					fyij = fs * d.y;

					d.z = rA_shared[wtx].z - rB_shared[j].z;

					fzij = fs * d.z;

					fA[wtx].v += (tested_type)(qB_shared[j] * vij);
					fA[wtx].x += (tested_type)(qB_shared[j] * fxij);
					fA[wtx].y += (tested_type)(qB_shared[j] * fyij);
					fA[wtx].z += (tested_type)(qB_shared[j] * fzij);
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
*/
}




#endif /* LAVA_KERNELS_H_ */
