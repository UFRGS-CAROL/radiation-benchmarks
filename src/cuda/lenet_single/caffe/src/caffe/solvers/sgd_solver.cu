#include <string>
#include <device_launch_parameters.h>

#include "caffe/util/gpu_math_functions.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Gtype, typename Wtype, typename Htype>
__global__ void SGDRegUpdateAllAndClear(int N,
  Gtype* g, Wtype* w, Htype* h,
    float momentum, float local_rate, float local_decay, bool reg_L2,  bool clear_grads) {
  CUDA_KERNEL_LOOP(i, N) {
    Wtype reg = reg_L2 ? w[i] : Wtype((Wtype(0) < w[i]) - (w[i] < Wtype(0)));
    Wtype gr = Wtype(g[i]) + reg * local_decay;
    gr = h[i] = momentum * h[i] + local_rate * gr;
    w[i] -= gr;
    g[i] = clear_grads ? Gtype(0) : Gtype(gr);
  }
}

template<>
__global__ void SGDRegUpdateAllAndClear<half, half, half>(int N,
  half* g, half* w, half* h,
    float momentum, float local_rate, float local_decay, bool reg_L2,  bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = __half2float(w[i]);
    float gf = __half2float(g[i]);
    float hf = __half2float(h[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gf += reg * local_decay;
    gf = hf = momentum * hf  + local_rate * gf;
    wf -= gf;

    h[i] = float2half_clip(hf);
    w[i] = float2half_clip(wf);
    g[i] = clear_grads ? hz : float2half_clip(gf);
  }
}

template<>
__global__ void SGDRegUpdateAllAndClear<float, float, half>(int N,
    float* g, float* w, half* h,
    float momentum, float local_rate, float local_decay, bool reg_L2,  bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float wf = w[i];
    float gf = g[i];
    float hf = __half2float(h[i]);

    float reg = reg_L2 ? wf : float((0.F < wf)-(wf < 0.F));
    gf += reg * local_decay;
    gf = hf = momentum * hf  + local_rate * gf;
    wf -= gf;

    h[i] = float2half_clip(hf);
    w[i] = wf;
    g[i] = clear_grads ? 0.F : gf;
  }
}

template<>
__global__ void SGDRegUpdateAllAndClear<half, float, float>(int N,
    half* g, float* w, float* h,
    float momentum, float local_rate, float local_decay, bool reg_L2, bool clear_grads) {
  half hz;
  CUDA_KERNEL_LOOP(i, N) {
    float reg = reg_L2 ? w[i] : (0.F < w[i]) - (w[i] < 0.F);
    float gr = __half2float(g[i]) + reg * local_decay;
    gr = h[i] = momentum * h[i] + local_rate * gr;
    w[i] -= gr;
    g[i] = clear_grads ? hz : float2half_clip(h[i]);
  }
}

template<typename Gtype, typename Wtype, typename Htype>
void sgd_reg_update_all_and_clear_gpu(int N,
  Gtype* g, Wtype* w, Htype* h,
  float momentum, float local_rate, const std::string& reg_type, float local_decay,
  void* handle,  bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));

  bool reg_L2 = (reg_type == "L2") || (reg_type == "L2_unitary");

  // NOLINT_NEXT_LINE(whitespace/operators)
  SGDRegUpdateAllAndClear<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N,
    g, w, h,
    momentum, local_rate, local_decay, reg_L2,  clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template void sgd_reg_update_all_and_clear_gpu<float16, double, double>(
    int, float16*, double*, double*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<float, float, float>(
    int, float*, float*, float*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<float, double, double>(
    int, float*, double*, double*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<float, float16, float16>(
    int, float*, float16*, float16*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<double, float, float>(
    int, double*, float*, float*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<double, double, double>(
    int, double*, double*, double*,
  float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<double, float16, float16>(
    int, double*, float16*, float16*,
  float, float, const std::string&, float,  void*, bool);

template void sgd_reg_update_all_and_clear_gpu<float, float, float16>(
    int, float*, float*, float16*,
    float, float, const std::string&, float,  void*, bool);
template void sgd_reg_update_all_and_clear_gpu<float, float, double>(
    int, float*, float*, double*,
    float, float, const std::string&, float,  void*, bool);

template<>
void
sgd_reg_update_all_and_clear_gpu<float16, float16>(int N,
  float16* g, float16* w, float16* h,
  float momentum, float local_rate, const std::string& reg_type, float local_decay,
  void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  SGDRegUpdateAllAndClear<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>> (N,
      reinterpret_cast<half*>(g), reinterpret_cast<half*>(w), reinterpret_cast<half*>(h),
      momentum, local_rate, local_decay, reg_type == "L2",  clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<>
void
sgd_reg_update_all_and_clear_gpu<float16, float>(int N,
    float16* g, float* w, float* h,
    float momentum,  float local_rate, const std::string& reg_type, float local_decay,
    void* handle, bool clear_grads) {
  cublasHandle_t cublas_handle =
      handle == nullptr ? Caffe::cublas_handle(0) : reinterpret_cast<cublasHandle_t>(handle);
  cudaStream_t stream;
  CUBLAS_CHECK(cublasGetStream(cublas_handle, &stream));
  // NOLINT_NEXT_LINE(whitespace/operators)
  SGDRegUpdateAllAndClear<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
      (N, reinterpret_cast<half*>(g), w, h, momentum, local_rate,
          local_decay, reg_type == "L2", clear_grads);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace caffe
