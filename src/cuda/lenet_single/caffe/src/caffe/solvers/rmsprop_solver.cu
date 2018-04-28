#include <string>
#include <device_launch_parameters.h>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"
template<typename Gtype, typename Wtype>
__global__ void RMSPropRegUpdateAllAndClear(int N,
    Gtype* g, Wtype *w, Wtype* h,
    float rms_decay, float delta, float local_rate, float local_decay, bool reg_L2,
    bool clear_grads) {
  CUDA_KERNEL_LOOP(i, N) {
    Wtype reg = reg_L2 ? w[i] : Wtype((Wtype(0) < w[i]) - (w[i] < Wtype(0)));
    Wtype gr = Wtype(g[i]) + reg * local_decay;
    Wtype hi = h[i] = rms_decay * h[i] + (1.F - rms_decay) * gr*gr;
    gr = local_rate * gr / (sqrt(hi) + delta);
    w[i] -= gr;
    g[i] = clear_grads ? Gtype(0) : Gtype(gr);
  }
}
#pragma clang diagnostic pop

template<>
__global__ void RMSPropRegUpdateAllAndClear<half, half>(int N,
    half* g, half* w, half* h,
    float rms_decay, float delta, float local_rate, float local_decay, bool reg_L2,
    bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gf += reg * local_decay;
    hf = rms_decay * hf + (1.F - rms_decay) * gf*gf;
    gf = local_rate * gf / (sqrt(hf) + delta);
    wf -= gf;

    h[i] = float2half_clip(hf);
    w[i] = float2half_clip(wf);
    g[i] = clear_grads ? hz : float2half_clip(gf);
  }
}


template<typename Gtype, typename Wtype>
void rmsprop_reg_update_and_clear_gpu(int N,
  Gtype* g, Wtype* w, Wtype* h,
  float rms_decay, float delta, float local_rate, const std::string& reg_type,
  float local_decay, void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  RMSPropRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
      g, w, h,
      rms_decay, delta, local_rate, local_decay, reg_type == "L2", clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void rmsprop_reg_update_and_clear_gpu<float16, float16>(int N,
  float16* g, float16* w, float16* h,
  float rms_decay, float delta, float local_rate, const std::string& reg_type,
  float local_decay, void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  RMSPropRegUpdateAllAndClear  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(N,
        reinterpret_cast<half*>(g), reinterpret_cast<half*>(w), reinterpret_cast<half*>(h),
        rms_decay, delta, local_rate, local_decay, reg_type == "L2", clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void rmsprop_reg_update_and_clear_gpu<float16, float>(int, float16*, float*, float*,
    float, float, float, const std::string&, float, void*, bool);
template void rmsprop_reg_update_and_clear_gpu<float16, double>(int, float16*, double*, double*,
    float, float, float, const std::string&, float, void*, bool);

template void rmsprop_reg_update_and_clear_gpu<float, float>(int, float*, float*, float*,
    float, float, float, const std::string&, float, void*, bool);
template void rmsprop_reg_update_and_clear_gpu<float, double>(int, float*, double*, double*,
    float, float, float, const std::string&, float, void*, bool);
template void rmsprop_reg_update_and_clear_gpu<float, float16>(int, float*, float16*, float16*,
    float, float, float, const std::string&, float, void*, bool);

template void rmsprop_reg_update_and_clear_gpu<double, float>(int, double*, float*, float*,
    float, float, float, const std::string&, float, void*, bool);
template void rmsprop_reg_update_and_clear_gpu<double, double>(int, double*, double*, double*,
    float, float, float, const std::string&, float, void*, bool);
template void rmsprop_reg_update_and_clear_gpu<double, float16>(int, double*, float16*, float16*,
    float, float, float, const std::string&, float, void*, bool);

}  // namespace caffe
