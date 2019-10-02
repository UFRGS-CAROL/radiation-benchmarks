/*
 * kernelCaller.h
 *
 *  Created on: Oct 2, 2019
 *      Author: carol
 */

#ifndef KERNELCALLER_H_
#define KERNELCALLER_H_

#include "nondmr_kernels.h"
#include "dmr_kernels.h"

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
struct KernelCaller {

	virtual ~KernelCaller() {
	}

	virtual void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu, FOUR_VECTOR<half_t>* d_fv_gpu_ht) = 0;
};

template<const uint32_t COUNT, const uint32_t THRESHOLD, typename half_t,
		typename real_t>
struct DMRKernelCaller: public KernelCaller<COUNT, THRESHOLD, half_t, real_t> {

	void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu, FOUR_VECTOR<half_t>* d_fv_gpu_ht) {
		kernel_gpu_cuda<COUNT, THRESHOLD> <<<blocks, threads, 0, stream.stream>>>(
				par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
				d_fv_gpu_ht);
	}
};

template<typename real_t>
struct UnhardenedKernelCaller: public KernelCaller<0, 0, real_t, real_t> {

	void kernel_call(dim3& blocks, dim3& threads, CudaStream& stream,
			par_str<real_t>& par_cpu, dim_str& dim_cpu, box_str* d_box_gpu,
			FOUR_VECTOR<real_t>* d_rv_gpu, real_t* d_qv_gpu,
			FOUR_VECTOR<real_t>* d_fv_gpu, FOUR_VECTOR<real_t>* d_fv_gpu_ht) {
		kernel_gpu_cuda<<<blocks, threads, 0, stream.stream>>>(par_cpu, dim_cpu,
				d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);
	}
};

#endif /* KERNELCALLER_H_ */
