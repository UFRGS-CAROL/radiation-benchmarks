// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TanHForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

template <typename Ftype, typename Btype>
void TanHLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data<Ftype>();
  Ftype* top_data = top[0]->mutable_gpu_data<Ftype>();
  const int count = bottom[0]->count();
  cudaStream_t stream = Caffe::thread_stream();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TanHForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Dtype>
__global__ void TanHBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    float tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (Dtype(1.) - tanhx * tanhx);
  }
}

template <typename Ftype, typename Btype>
void TanHLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* top_data = top[0]->gpu_data<Btype>();
    const Btype* top_diff = top[0]->gpu_diff<Btype>();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff<Btype>();
    const int count = bottom[0]->count();
    cudaStream_t stream = Caffe::thread_stream();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TanHBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_FB(TanHLayer);

}  // namespace caffe
