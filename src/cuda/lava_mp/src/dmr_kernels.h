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

//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------
// #if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
__global__ void kernel_gpu_cuda_block_check(par_str<real_t> d_par_gpu, dim_str d_dim_gpu,
		box_str* d_box_gpu, FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
		FOUR_VECTOR<real_t>* d_fv_gpu_rt, FOUR_VECTOR<half_t>* d_fv_gpu_ht) {

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

					u2_rt = a2_rt * r2_rt;

					//DMR
					u2_ht = a2_ht * r2_ht;

					vij_rt = exp__(-u2_rt);

					//DMR
					vij_ht = exp__(-u2_ht);

					fs_rt = real_t(2.0) * vij_rt;

					//DMR
					fs_ht = half_t(2.0) * vij_ht;

					d_rt.x = rA_shared_rt[wtx].x - rB_shared_rt[j].x;

					//DMR
					d_ht.x = rA_shared_ht[wtx].x - rB_shared_ht[j].x;

					fxij_rt = fs_rt * d_rt.x;

					//DMR
					fxij_ht = fs_ht * d_ht.x;

					d_rt.y = rA_shared_rt[wtx].y - rB_shared_rt[j].y;

					//DMR
					d_ht.y = rA_shared_ht[wtx].y - rB_shared_ht[j].y;

					fyij_rt = fs_rt * d_rt.y;

					//DMR
					fyij_ht = fs_ht * d_ht.y;

					d_rt.z = rA_shared_rt[wtx].z - rB_shared_rt[j].z;

					//DMR
					d_ht.z = rA_shared_ht[wtx].z - rB_shared_ht[j].z;

					fzij_rt = fs_rt * d_rt.z;

					//DMR
					fzij_ht = fs_ht * d_ht.z;

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
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(fA_ht[wtx], fA_rt[wtx]);
						fA_ht[wtx] = fA_rt[wtx];
					}
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
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
	}
}

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
__global__ void kernel_gpu_cuda(par_str<real_t> d_par_gpu, dim_str d_dim_gpu,
		box_str* d_box_gpu, FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
		FOUR_VECTOR<real_t>* d_fv_gpu_rt, FOUR_VECTOR<half_t>* d_fv_gpu_ht) {

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
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(r2_ht, r2_rt);
					}
					//-----------------------------------------------------

					u2_rt = a2_rt * r2_rt;

					//DMR
					u2_ht = a2_ht * r2_ht;

					//CHECK bit--------------------------------------------
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(u2_ht, u2_rt);
					}
					//-----------------------------------------------------

					vij_rt = exp__(-u2_rt);

					//DMR
					vij_ht = exp__(-u2_ht);

					//CHECK bit--------------------------------------------
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(vij_ht, vij_rt);
					}
					//-----------------------------------------------------


					fs_rt = real_t(2.0) * vij_rt;

					//DMR
					fs_ht = half_t(2.0) * vij_ht;

					//CHECK bit--------------------------------------------
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(fs_ht, fs_rt);
					}
					//-----------------------------------------------------

					d_rt.x = rA_shared_rt[wtx].x - rB_shared_rt[j].x;

					//DMR
					d_ht.x = rA_shared_ht[wtx].x - rB_shared_ht[j].x;

					//CHECK bit--------------------------------------------
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(d_ht.x, d_rt.x);
					}
					//-----------------------------------------------------

					fxij_rt = fs_rt * d_rt.x;

					//DMR
					fxij_ht = fs_ht * d_ht.x;

					d_rt.y = rA_shared_rt[wtx].y - rB_shared_rt[j].y;

					//DMR
					d_ht.y = rA_shared_ht[wtx].y - rB_shared_ht[j].y;

					fyij_rt = fs_rt * d_rt.y;

					//DMR
					fyij_ht = fs_ht * d_ht.y;

					d_rt.z = rA_shared_rt[wtx].z - rB_shared_rt[j].z;

					//DMR
					d_ht.z = rA_shared_ht[wtx].z - rB_shared_ht[j].z;

					fzij_rt = fs_rt * d_rt.z;

					//DMR
					fzij_ht = fs_ht * d_ht.z;

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
					if ((j % COUNT) == 0) {
						check_bit_error<THRESHOLD>(fA_ht[wtx], fA_rt[wtx]);
						fA_ht[wtx] = fA_rt[wtx];
					}
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
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
	}
}


#endif /* DMR_KERNELS_H_ */
