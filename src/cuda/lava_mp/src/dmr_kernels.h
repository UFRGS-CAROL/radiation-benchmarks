/*
 * nondmr_kernels.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef DMR_KERNELS_H_
#define DMR_KERNELS_H_

#include "device_funtions.h"
#include "types.h"

template<const uint32_t COUNT, typename half_t, typename real_t>
__global__ void kernel_gpu_cuda_dmr(par_str<real_t> d_par_gpu,
		dim_str d_dim_gpu, box_str* d_box_gpu, FOUR_VECTOR<real_t>* d_rv_gpu,
		real_t* d_qv_gpu, FOUR_VECTOR<real_t>* d_fv_gpu_rt,
		FOUR_VECTOR<half_t>* d_fv_gpu_ht, uint32_t THRESHOLD) {

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
		half_t a2_half = half_t(a2);

		// home box
		int first_i;
		FOUR_VECTOR<real_t> *rA;
		FOUR_VECTOR<real_t> *fA;
		FOUR_VECTOR<half_t> *fA_half;

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
		fA = &d_fv_gpu_rt[first_i];
		fA_half = &d_fv_gpu_ht[first_i];
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

					r2 = rA_shared[wtx].v + rB_shared[j].v-
					DOT(
							rA_shared[wtx],
							rB_shared[j]
					);

					//----DMR----------
					FOUR_VECTOR<half_t> rA_half, rB_half;
					rA_half = rA_shared[wtx];
					rB_half = rB_shared[j];
					half_t r2_half_t = rA_half.v + rB_half.v-
					DOT(
							rA_half,
							rB_half
					);
					//-----------------

					u2 = a2 * r2;
					//----DMR----------
					half_t u2_half = a2_half * r2_half_t;
					//-----------------

					vij = exp__(-u2);
					//----DMR----------
					half_t vij_half = exp__(-u2_half);
					//-----------------

					fs = real_t(2.0) * vij;
					//----DMR----------
					half_t fs_half = half_t(2.0) * vij_half;
					//-----------------

					//----DMR----------
					THREE_VECTOR<half_t> d_half;
					half_t fxij_half, fyij_half, fzij_half;
					//-----------------

					d.x = rA_shared[wtx].x - rB_shared[j].x;

					//----DMR----------
					d_half.x = rA_half.x - rB_half.x;
					//-----------------

					fxij = fs * d.x;
					//----DMR----------
					fxij_half = fs_half * d_half.x;
					//-----------------

					d.y = rA_shared[wtx].y - rB_shared[j].y;
					//----DMR----------
					d_half.y = rA_half.y - rB_half.y;
					//-----------------

					fyij = fs * d.y;
					//----DMR----------
					fyij_half = fs_half * d_half.y;
					//-----------------

					d.z = rA_shared[wtx].z - rB_shared[j].z;
					//----DMR----------
					d_half.z = rA_half.z - rB_half.z;
					//-----------------

					fzij = fs * d.z;
					//----DMR----------
					fzij_half = fs_half * d_half.z;
					//-----------------

					fA[wtx].v += (real_t) (qB_shared[j] * vij);
					fA[wtx].x += (real_t) (qB_shared[j] * fxij);
					fA[wtx].y += (real_t) (qB_shared[j] * fyij);
					fA[wtx].z += (real_t) (qB_shared[j] * fzij);

					//----DMR----------
					half_t qB_shared_half = qB_shared[j];
//					fA_half[wtx].v += (qB_shared_half * vij_half);
//					fA_half[wtx].x += (qB_shared_half * fxij_half);
//					fA_half[wtx].y += (qB_shared_half * fyij_half);
//					fA_half[wtx].z += (qB_shared_half * fzij_half);
					FOUR_VECTOR<half_t> tmp;
					tmp = fA[wtx];
					tmp.v += (qB_shared_half * vij_half);
					tmp.x += (qB_shared_half * fxij_half);
					tmp.y += (qB_shared_half * fyij_half);
					tmp.z += (qB_shared_half * fzij_half);
					if ((j % (COUNT + 1)) == 0) {
						check_bit_error(tmp, fA[wtx], THRESHOLD);
					}
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

/**
 * 	//---------------------------------------------------------------------
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
 real_t a2_rt = real_t(2.0) * d_par_gpu.alpha * d_par_gpu.alpha;

 //mixed dmr
 half_t a2_ht = half_t(a2_rt);

 // home box
 int first_i;
 FOUR_VECTOR<real_t> *rA_rt;
 FOUR_VECTOR<real_t> *fA_rt;
 __shared__ FOUR_VECTOR<real_t> rA_shared_rt[200];

 //DMR
 FOUR_VECTOR<half_t> *fA_ht;
 __shared__ FOUR_VECTOR<half_t> rA_shared_ht[200];

 // nei box
 int pointer;
 int k = 0;
 int first_j;
 FOUR_VECTOR<real_t> *rB_rt;
 real_t *qB_rt;
 int j = 0;
 __shared__ FOUR_VECTOR<real_t> rB_shared_rt[200];
 __shared__ real_t qB_shared_rt[200];

 //DMR
 __shared__ FOUR_VECTOR<half_t> rB_shared_ht[200];
 __shared__ half_t qB_shared_ht[200];

 // common
 real_t r2_rt;
 real_t u2_rt;
 real_t vij_rt;
 real_t fs_rt;
 real_t fxij_rt;
 real_t fyij_rt;
 real_t fzij_rt;
 THREE_VECTOR<real_t> d_rt;

 //DMR
 half_t r2_ht;
 half_t u2_ht;
 half_t vij_ht;
 half_t fs_ht;
 half_t fxij_ht;
 half_t fyij_ht;
 half_t fzij_ht;
 THREE_VECTOR<half_t> d_ht;

 //-------------------------------------------------------------
 //	Home box
 //-------------------------------------------------------------

 //-------------------------------------------------------------
 //	Setup parameters
 //-------------------------------------------------------------

 // home box - box parameters
 first_i = d_box_gpu[bx].offset;

 // home box - distance, force, charge and type parameters
 rA_rt = &d_rv_gpu[first_i];
 fA_rt = &d_fv_gpu_rt[first_i];

 //DMR
 fA_ht = &d_fv_gpu_ht[first_i];

 //-------------------------------------------------------------
 //	Copy to shared memory
 //-------------------------------------------------------------

 // home box - shared memory
 while (wtx < NUMBER_PAR_PER_BOX) {
 rA_shared_rt[wtx] = rA_rt[wtx];
 //DMR
 rA_shared_ht[wtx] = rA_shared_rt[wtx];

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
 rB_rt = &d_rv_gpu[first_j];
 qB_rt = &d_qv_gpu[first_j];

 //-----------------------------------------------------
 //	Setup parameters
 //-----------------------------------------------------

 // nei box - shared memory
 while (wtx < NUMBER_PAR_PER_BOX) {
 rB_shared_rt[wtx] = rB_rt[wtx];
 qB_shared_rt[wtx] = qB_rt[wtx];

 //DMR
 rB_shared_ht[wtx] = rB_shared_rt[wtx];
 qB_shared_ht[wtx] = half_t(qB_shared_rt[wtx]);

 wtx = wtx + NUMBER_THREADS;
 }
 wtx = tx;

 // synchronize threads because in next section each thread accesses data brought in by different threads here
 __syncthreads();

 //-----------------------------------------------------
 //	Calculation
 //-----------------------------------------------------
 while (wtx < NUMBER_PAR_PER_BOX) {

 #pragma unroll
 for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

 r2_rt =
 rA_shared_rt[wtx].v
 + rB_shared_rt[j].v- DOT(rA_shared_rt[wtx], rB_shared_rt[j]);

 //DMR
 r2_ht =
 rA_shared_ht[wtx].v
 + rB_shared_ht[j].v- DOT(rA_shared_ht[wtx], rB_shared_ht[j]);

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(r2_ht, r2_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 u2_rt = a2_rt * r2_rt;

 //DMR
 u2_ht = a2_ht * r2_ht;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(u2_ht, u2_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 vij_rt = exp__(-u2_rt);

 //DMR
 vij_ht = exp__(-u2_ht);

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(vij_ht, vij_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 fs_rt = real_t(2.0) * vij_rt;

 //DMR
 fs_ht = half_t(2.0) * vij_ht;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(fs_ht, fs_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 d_rt.x = rA_shared_rt[wtx].x - rB_shared_rt[j].x;

 //DMR
 d_ht.x = rA_shared_ht[wtx].x - rB_shared_ht[j].x;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(d_ht.x, d_rt.x, THRESHOLD);
 //					}
 //-----------------------------------------------------

 fxij_rt = fs_rt * d_rt.x;

 //DMR
 fxij_ht = fs_ht * d_ht.x;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(fxij_ht, fxij_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 d_rt.y = rA_shared_rt[wtx].y - rB_shared_rt[j].y;

 //DMR
 d_ht.y = rA_shared_ht[wtx].y - rB_shared_ht[j].y;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(d_ht.y, d_rt.y, THRESHOLD);
 //					}
 //-----------------------------------------------------

 fyij_rt = fs_rt * d_rt.y;

 //DMR
 fyij_ht = fs_ht * d_ht.y;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(fyij_ht, fyij_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 d_rt.z = rA_shared_rt[wtx].z - rB_shared_rt[j].z;

 //DMR
 d_ht.z = rA_shared_ht[wtx].z - rB_shared_ht[j].z;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(d_ht.z, d_rt.z, THRESHOLD);
 //					}
 //-----------------------------------------------------

 fzij_rt = fs_rt * d_rt.z;

 //DMR
 fzij_ht = fs_ht * d_ht.z;

 //CHECK bit--------------------------------------------
 //					if ((j % (COUNT + 1)) == 0) {
 //						check_bit_error(fzij_ht, fzij_rt, THRESHOLD);
 //					}
 //-----------------------------------------------------

 fA_rt[wtx].v += real_t(qB_shared_rt[j] * vij_rt);
 fA_rt[wtx].x += real_t(qB_shared_rt[j] * fxij_rt);
 fA_rt[wtx].y += real_t(qB_shared_rt[j] * fyij_rt);
 fA_rt[wtx].z += real_t(qB_shared_rt[j] * fzij_rt);

 //DMR
 fA_ht[wtx].v += half_t(qB_shared_ht[j] * vij_ht);
 fA_ht[wtx].x += half_t(qB_shared_ht[j] * fxij_ht);
 fA_ht[wtx].y += half_t(qB_shared_ht[j] * fyij_ht);
 fA_ht[wtx].z += half_t(qB_shared_ht[j] * fzij_ht);

 //------------------------------------------------
 //DMR CHECKING
 //					if (((j +1) % COUNT) == 0) {
 //						printf("passou count\n");
 //						check_bit_error(fA_ht[wtx], fA_rt[wtx], THRESHOLD);
 //					}
 //------------------------------------------------
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

 //		check_bit_error(fA_ht[tx], fA_rt[tx], THRESHOLD);

 //------------------------------------------------------------------------------------------------------------------------------------------------------160
 //	nei box loop END
 //------------------------------------------------------------------------------------------------------------------------------------------------------160
 }
 */

#endif /* DMR_KERNELS_H_ */
